# Training And Test Log

## Revalidation

Revalidated on 2026-07-19 with the project's `agent0-gpu` Python 3.12.13
environment:

```text
/home/share/huadjyin/home/s_qinhua2/01software/miniconda3/envs/agent0-gpu/bin/python -m unittest discover -s tests -q
----------------------------------------------------------------------
Ran 170 tests in 5.097s

OK
```

Raw four-rank training logs for rounds 1-3 are stored under
`outputs/candidate_min3/pilot3/launcher_logs/`. Raw fixed-evaluation shard logs
are stored beside each shard JSON under
`evaluations/candidate_coevolution_min3/pilot3/`.

## Completed Checks

- Candidate/compiler, DCA reward, and recovery contract suite: 16 passed.
- Python compile audit: passed for candidate modules and all new train/eval runners.
- Skill-identifiability gate: passed T1-T4; Full-vs-ablation utility gaps were
  1.5163, 1.5532, 0.1139, and 1.3970.
- Six-source warm-start audit: 20 states; compiler/reference/Teacher recall and
  permutation consistency 100%; semantic duplicate/conflict 0%.
- Semantic split audit: train/dev overlap 0.
- Three-round pilot: completed 3/3 rounds; 0 checkpoints promoted.

## Warm Start

| Stage | Objective | Steps | LR | Train loss |
|---|---|---:|---:|---:|
| A1 | family | 4 | 1e-5 | 5.2687 |
| A2 | listwise | 4 | 5e-6 | 6.0304 |
| A3 | preference | 4 | 2e-6 | 2.8351 |

A1, A2, and A3 made identical decisions on the six-state held-out development
set: top-1 0%, family accuracy 33.3%, mean Teacher regret 1.4167, active probe
33.3%, and Memory 0%. This confirms that the warm start broke fixed Observe but
did not establish useful defense ranking.

The cluster's old kernel can leave the `torchrun` elastic parent waiting after
all NCCL workers have destroyed their process groups. The branch therefore
includes `scripts/launch_candidate_ddp.py`, which launches rank processes
directly and stores one log per rank. Parallel evaluations also use isolated
Triton cache directories to prevent compilation-cache races.
