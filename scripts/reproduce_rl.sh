#!/bin/bash
# Task 2 / RL study, in the order the reported results were produced (10K users = all of Yambda-50M).
# GPU recommended (RTX 5070 Ti: ~1.5 h in total). Jobs run one at a time; each 10K-user job peaks at ~10-15 GB RAM.
set -e
cd "$(dirname "$0")/../rl_experiments"
mkdir -p logs
uv run python organic_analysis.py                                    # organic vs pushed listens, artist-return signal
uv run python tune_baselines.py --models decaypop,itemknn,covisit,als # tune on 10K val, then one test run each
uv run python train_seq.py --n 10000 --stage val --model sasrec --steps 12000 --B 256 --d 128 --eval_at 3000,6000,9000,12000
uv run python train_seq.py --n 10000 --stage test --model sasrec --steps 9000 --B 256 --d 128 --seed 0 --tag _s0
uv run python rl_push.py --n 10000 --stage test --ref sasrec_s0 --aux_steps 3000 --steps 300
uv run python rm_push_quality.py --n 10000 --stage test
uv run python summarize_v3.py                                        # -> out/tables_v3.md
