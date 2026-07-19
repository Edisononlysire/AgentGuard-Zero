#!/usr/bin/env python3
"""Fail-fast host, environment, model and input checks for the A100 Agent0 job."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import jinja2
import psutil
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


EXPECTED = {
    "torch": "2.10.0",
    "vllm": "0.18.0",
    "transformers": "4.57.1",
    "ray": "2.54.0",
    "tensordict": "0.13.0",
    "torchdata": "0.11.0",
    "Flask": "3.1.1",
    "mathruler": "0.1.0",
    "stopit": "1.1.2",
    "codetiming": "1.4.0",
    "pylatexenc": "2.10",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def check_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            fail(f"Required localhost port {port} is unavailable: {exc}")


def check_native_compiler() -> None:
    """Exercise the same inherited environment Triton uses for C extensions."""
    compiler = os.environ.get("CC")
    if not compiler:
        fail("CC is unset; Triton would select an uncontrolled compiler")
    compiler_path = Path(compiler).resolve()
    if not compiler_path.is_file() or not os.access(compiler_path, os.X_OK):
        fail(f"CC is not executable: {compiler_path}")
    with tempfile.TemporaryDirectory(prefix="agent0_cc_") as tmpdir:
        source = Path(tmpdir) / "preflight.c"
        library = Path(tmpdir) / "preflight.so"
        source.write_text("int agent0_preflight(void) { return 42; }\n", encoding="utf-8")
        result = subprocess.run(
            [str(compiler_path), str(source), "-O2", "-shared", "-fPIC", "-o", str(library)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode or not library.is_file():
            details = (result.stderr or result.stdout).strip()
            fail(f"Native compiler preflight failed with {compiler_path}: {details}")


def check_jsonl(path: Path, prompt_key: str, answer_key: str) -> int:
    count = 0
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get(prompt_key), str) or not isinstance(record.get(answer_key), str):
                fail(f"{path}:{lineno} must contain string keys {prompt_key!r} and {answer_key!r}")
            count += 1
    if count == 0:
        fail(f"Dataset is empty: {path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--questioner-template", required=True)
    parser.add_argument("--qwen-template", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--expected-gpus", type=int, default=4)
    args = parser.parse_args()

    venv = Path(args.venv).resolve()
    if Path(sys.prefix).resolve() != venv:
        fail(f"Wrong Python prefix: expected {venv}, got {sys.prefix}")
    if site := os.environ.get("PYTHONPATH"):
        allowed = str(Path(args.questioner_template).resolve().parents[2])
        if site != allowed:
            fail(f"Unexpected PYTHONPATH={site!r}; expected only {allowed!r}")
    if not sys.flags.no_user_site:
        fail("User site-packages are enabled; set PYTHONNOUSERSITE=1 and use python -s")

    for package, expected in EXPECTED.items():
        actual = metadata.version(package)
        if actual != expected:
            fail(f"{package} version mismatch: expected {expected}, got {actual}")

    check_native_compiler()

    import ray
    import tensordict
    import transformers
    import vllm

    paths = {
        "transformers": Path(transformers.__file__).resolve(),
        "tensordict": Path(tensordict.__file__).resolve(),
    }
    for name, path in paths.items():
        if venv not in path.parents:
            fail(f"{name} is not loaded from the overlay: {path}")
    for name, module in (("torch", torch), ("vllm", vllm), ("ray", ray)):
        path = Path(module.__file__).resolve()
        if ".local" in path.parts:
            fail(f"{name} leaked from user site-packages: {path}")

    arches = torch.cuda.get_arch_list()
    if "sm_80" not in arches:
        fail(f"Torch does not contain A100/sm_80 kernels: {arches}")
    if not torch.cuda.is_available():
        fail("CUDA is unavailable")
    if torch.cuda.device_count() != args.expected_gpus:
        fail(f"Expected {args.expected_gpus} visible GPUs, got {torch.cuda.device_count()}")
    gpu_lines = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        if props.major != 8 or props.minor != 0 or "A100" not in props.name:
            fail(f"GPU {index} is not an A100/sm80: {props.name}, capability={props.major}.{props.minor}")
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 38 * 1024**3:
            fail(f"GPU {index} has only {free_bytes / 1024**3:.2f} GiB free")
        gpu_lines.append(f"gpu{index}={props.name}:{free_bytes / 1024**3:.2f}GiB-free")

    shm = psutil.disk_usage("/dev/shm")
    if shm.free < 64 * 1024**3:
        fail(f"/dev/shm has only {shm.free / 1024**3:.2f} GiB free")
    memory = psutil.virtual_memory()
    min_available_ram_gib = int(os.getenv("AGENT0_MIN_AVAILABLE_RAM_GIB", "130"))
    if memory.available < min_available_ram_gib * 1024**3:
        fail(f"Host has only {memory.available / 1024**3:.2f} GiB available RAM")
    cgroup_limit_path = Path("/sys/fs/cgroup/memory.max")
    if cgroup_limit_path.is_file():
        raw_limit = cgroup_limit_path.read_text(encoding="utf-8").strip()
        if raw_limit != "max" and int(raw_limit) < min_available_ram_gib * 1024**3:
            fail(f"Job cgroup memory limit is only {int(raw_limit) / 1024**3:.2f} GiB")
    storage = Path(args.storage)
    storage.mkdir(parents=True, exist_ok=True)
    disk = psutil.disk_usage(storage)
    if disk.free < 120 * 1024**3:
        fail(f"Storage has only {disk.free / 1024**3:.2f} GiB free; two checkpoints need substantial space")

    for port in (5000, 5001, 8080):
        check_port_free(port)

    model = Path(args.model).resolve()
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    missing_shards = [name for name in shard_names if not (model / name).is_file()]
    if missing_shards:
        fail(f"Missing model shards: {missing_shards}")
    tensor_names: set[str] = set()
    parameter_count = 0
    for shard_name in shard_names:
        with safe_open(model / shard_name, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            tensor_names.update(keys)
            parameter_count += sum(math.prod(handle.get_slice(key).get_shape()) for key in keys)
    if tensor_names != set(index["weight_map"]):
        fail("Safetensors keys do not match model.safetensors.index.json")

    config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    if config.model_type != "qwen3" or not tokenizer.chat_template:
        fail(f"Unexpected model/tokenizer: model_type={config.model_type}, chat_template={bool(tokenizer.chat_template)}")
    if config.num_attention_heads % 2 or config.num_key_value_heads % 2:
        fail("Qwen3 attention heads are incompatible with tensor_parallel_size=2")
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "Generate one math problem."}],
        tokenize=True,
        add_generation_prompt=True,
    )

    jinja = jinja2.Environment()
    for template_path in (Path(args.questioner_template), Path(args.qwen_template)):
        jinja.parse(template_path.read_text(encoding="utf-8"))

    tokens = json.loads(Path(args.tokens).read_text(encoding="utf-8"))
    if not all(key in tokens and isinstance(tokens[key], str) for key in ("huggingface", "wandb")):
        fail("tokens.json must contain string keys 'huggingface' and 'wandb'")
    train_count = check_jsonl(Path(args.train), "problem", "answer")
    val_count = check_jsonl(Path(args.val), "problem", "answer")

    print("Agent0 host preflight OK")
    print("versions=" + ",".join(f"{key}:{value}" for key, value in EXPECTED.items()))
    print("gpus=" + ",".join(gpu_lines))
    print(
        f"model_parameters={parameter_count} shards={len(shard_names)} "
        f"train_rows={train_count} val_rows={val_count} "
        f"ram_free_gib={memory.available / 1024**3:.2f} "
        f"shm_free_gib={shm.free / 1024**3:.2f} disk_free_gib={disk.free / 1024**3:.2f}"
    )


if __name__ == "__main__":
    main()
