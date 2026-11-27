#!/usr/bin/env bash
# Must finish in < 10 min and rerun a SMALL version of your pipeline end to end. Adapt it at every milestone.
# Run it with the project environment: `uv run bash run_smoke.sh`, or activate .venv first (README.md, Setup).
# Windows: Git Bash (comes with Git for Windows) or WSL, or paste the lines below into PowerShell one by one.
# So that it works everywhere, write plain `python -m ...` lines here: no bash-only syntax (export, $(...), &&-chains).
# Every output goes to results/smoke/ (git-ignored): the smoke run never adds rows to results/results.jsonl.
set -euo pipefail
python -m pytest -q tests/public
# Day 1: n-gram perplexity on the digit grammar (seconds)
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part lm --lm_sizes 50 500 --max_order 3 --out results/smoke --jsonl results/smoke/results.jsonl
# Day 2: one model on 3 LibriSpeech utterances, then CTC decoding (greedy / beam / beam + bigram) on synthetic posteriors
python -m speech.asr_benchmark --librispeech data/LibriSpeech --n 3 --models facebook/wav2vec2-base-960h --batch 1 --device cpu --out results/smoke/asr.jsonl
python -m speech.ctc_lm --selftest --out results/smoke/ctc
# Day 3: LocalAgreement and the clock on the mock recogniser, then our deployed system on 2 utterances fed in real time
python -m speech.streaming_sim --selftest
python -m asr_demo.replay --librispeech data/LibriSpeech --n 2 --realtime --backends reference/fastconformer-114M-int8 --jsonl results/smoke/replay.jsonl
