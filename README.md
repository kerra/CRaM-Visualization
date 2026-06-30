# Qwen + CrAM Interactive Explainer

This repository combines the official CrAM attention modification with an
interactive Qwen visualization and a live audience misinformation demo.

The retained pipeline supports:

- influential-head scoring and selection for `Qwen/Qwen1.5-0.5B-Chat`;
- the official `Re_Weighting_Strategy.edit_attention_mask` CrAM calculation;
- a presenter and phone audience workflow;
- side-by-side Vanilla and CrAM Qwen traces.

## Installation and usage

See [`transformer-explainer/QWEN_CRAM_SETUP.md`](transformer-explainer/QWEN_CRAM_SETUP.md)
for installation, head generation, startup commands, architecture notes, and
troubleshooting.

Install all Python dependencies from the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Install the browser application:

```bash
cd transformer-explainer
npm ci
```

## Core retained files

- `utils/re_weighting.py`: official CrAM calculation and head-effect scoring.
- `utils/find_best_heads.py`: aggregate and select influential heads.
- `utils/qwen2_cram_patch.py`: Qwen per-head attention-mask compatibility.
- `datasets/nq_0.5b/qwen/`: generated Qwen head scores and selected heads.
- `nq_100_qwen05.json`: input examples for regenerating the head mapping.
- `curated_short_conflict_trivia.json`: audience demonstration questions.
- `transformer-explainer/`: Qwen trace server and interactive frontend.

The original CrAM paper is available at
[arXiv:2406.11497](https://arxiv.org/abs/2406.11497).
