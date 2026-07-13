# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, StoppingCriteriaList

from verl import DataProto
from agentguard_zero.json_stopping import CompleteJSONObjectCriteria
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer

    def activate(self, device) -> None:
        """Move an inference-only rollout replica onto its local accelerator."""
        if isinstance(device, int):
            device = torch.device(get_device_name(), device)
        self.module.to(device)
        actual_device = next(self.module.parameters()).device
        if actual_device != device:
            raise RuntimeError(f"HF rollout replica stayed on {actual_device}; expected {device}")
        self.module.eval()

    def deactivate(self) -> None:
        """Release replica GPU memory before reward-model work starts."""
        self.module.to("cpu")
        self.module.eval()
        get_torch_device().empty_cache()

    def _model_type(self) -> str:
        candidates = [
            self.module,
            getattr(self.module, "module", None),
            getattr(self.module, "_fsdp_wrapped_module", None),
        ]
        for candidate in candidates:
            config = getattr(candidate, "config", None) if candidate is not None else None
            model_type = getattr(config, "model_type", "")
            if model_type:
                return str(model_type)
        return ""

    def _special_token_id(self, name: str, value):
        if value is not None:
            return value
        candidates = [
            self.module,
            getattr(self.module, "module", None),
            getattr(self.module, "_fsdp_wrapped_module", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            for config in (
                getattr(candidate, "generation_config", None),
                getattr(candidate, "config", None),
                getattr(getattr(candidate, "config", None), "text_config", None),
            ):
                token_id = getattr(config, name, None) if config is not None else None
                if token_id is not None:
                    return token_id
        return None

    @contextlib.contextmanager
    def update_sampling_params(self, **kwargs):
        """Temporarily apply vLLM-style sampling params for agent rollouts."""
        key_map = {
            "max_tokens": "response_length",
            "max_new_tokens": "response_length",
        }
        supported_keys = {"do_sample", "temperature", "top_p", "top_k", "n", "response_length"}
        old_values = {}

        for key, value in kwargs.items():
            mapped_key = key_map.get(key, key)
            if value is None or mapped_key not in supported_keys:
                continue
            if mapped_key not in old_values:
                old_values[mapped_key] = self.config.get(mapped_key, None)
            self.config[mapped_key] = value

        try:
            yield
        finally:
            for key, value in old_values.items():
                self.config[key] = value

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # The verl-tool trainer repeats prompts by rollout.n before calling
            # this backend. Generate one continuation per repeated prompt to
            # avoid applying n twice.
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)
        stopping_criteria = None
        if self.config.get("stop_on_complete_json", False):
            if self.tokenizer is None:
                raise RuntimeError("JSON completion stopping requires a tokenizer")
            stopping_criteria = StoppingCriteriaList(
                [CompleteJSONObjectCriteria(self.tokenizer, batch_size=prompts.batch.batch_size[0])]
            )

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = self._special_token_id("eos_token_id", prompts.meta_info.get("eos_token_id"))
        pad_token_id = self._special_token_id("pad_token_id", prompts.meta_info.get("pad_token_id"))
        if pad_token_id is None:
            pad_token_id = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id
        if eos_token_id is None or pad_token_id is None:
            raise ValueError("HF rollout requires a resolvable EOS and padding token id")

        was_training = self.module.training
        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # Qwen3.5 has nested layer FSDP wrappers. Keeping those wrappers
            # sharded makes every decode token repeat layer all-gathers.
            recurse = self._model_type() == "qwen3_5"
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=recurse)
        generation_inputs = {
            "input_ids": idx,
            "attention_mask": attention_mask,
        }
        if self._model_type() != "qwen3_5":
            generation_inputs["position_ids"] = position_ids

        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                **generation_inputs,
                do_sample=do_sample,
                max_new_tokens=response_length,
                synced_gpus=torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + int(response_length)
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.full(
                size=(generated_batch_size, delta_length),
                fill_value=int(pad_token_id),
                device=seq.device,
                dtype=seq.dtype,
            )
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        if position_ids.dim() == 3:
            # Multimodal RoPE is stored batch-first as (bs, channels, seq).
            # Every text response channel advances by the same token offset.
            delta_position_id = delta_position_id.view(1, 1, -1)
            response_position_ids = position_ids[:, :, -1:] + delta_position_id
        else:
            delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
            response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train(was_training)
        return DataProto(batch=batch)
