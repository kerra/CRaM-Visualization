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

# Note

The auditory can use the QR code and input their statements only outside of private network. So probably will not under corporate or uni network, I ran it on mobile net + hotspot

The questions displayed as well as associated documents are in curated_short_conflict_trivia.json file. You can modify it or create another file with questions and documents as well as score, but be sure to keep the structure same.

- Currently it is designed in such a way, that 3 hardcoded questions are displayed on presentation screen (the ones I found most interesting among Trivia subset).
- The audience can provide own sentence in order to confuse the model. If nothing provided the pipeline takes docs with low credibility from json.
- For the explainer screen only 3 inputs are used, otherwise everything will be so small, that we hardly see difference.
- In the explainer you will have 2 options through each you can navigate and see how the model went through calculations: Vanilla (not-modified model) and CRaM (with method pluggged in)
- Since the CRaM method modifies not each head, when expending attention block of the particular head you either see as the end calculation Softmax with notion that this head is not selected, or actual CRaM if the head was selected

## Compatibility with other models
This repo is designed only on presentation purposes for Qwen/Qwen1.5-0.5B-Chat model. I cannot guarantee it will work on another Qwen model.
If one may want to run it on another model, the best way is to:
1. Clone official [CRaM repo](https://github.com/Aatrox103/CrAM)
2. Run the pipeline for head selection
   - Generate the small subset of main dataset
      While being in parent CRaM folder run in cmd 
     ```
         jq '.[0:100] | map(.ori_fake = [.ori_fake[0]])' \
      nq_1000_bge.json > nq_100_{your_model}.json
     ```
    - Then generate per-head scores for selected model
    ```
        PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -c "
    from utils.find_best_heads import casual_tracing_per_head
    
    casual_tracing_per_head(
        filepath='nq_100_{your_model}.json',
        LLM='{your_model}t', # hf link 
        output_dir='datasets/nq_0.5b',
        contexts_type=['ori_fake', 'reranked_dense'],
        topk=4,
        )
    "
    ```

    This will create datasets/nq_0.5b/{your_model}/heads_scores.json. It runs 385 forward passes per example—one baseline plus 24 × 16 heads, so 100 examples can take several hours. It saves after every example; if interrupted with Ctrl+C, rerunning the same command resumes.

    - Aggregate the scores
    ```
          .venv/bin/python -c "
      from utils.find_best_heads import casual_tracing_combine_all
      casual_tracing_combine_all('datasets/nq_0.5b/{your_model}')
    "
    ```
    This will create datasets/nq_0.5b/{your_model}/heads_scores_mean.json

   - Select the influential heads. 64 is a reasonable initial candidate from 384 total heads
    ```
        .venv/bin/python -c "
    from utils.find_best_heads import find_top_k_heads
    find_top_k_heads('datasets/nq_0.5b/qwen', topk=64)
    "
    ```
    So you will have selected_heads.json that will be used in explainer.

  3. Modify the needed parts in /src and trace_backed for the full pipeline to be actually compatible with the model or do it from scratch without cram on [official transformer-explainer repo](https://github.com/poloclub/transformer-explainer)
