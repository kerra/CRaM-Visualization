# Qwen + CrAM Explainer

This document describes the combined project rooted at:

```text
CrAM/
├── datasets/                       # Qwen influential-head data
├── utils/                          # original CrAM implementation
│   ├── re_weighting.py
│   ├── find_best_heads.py
│   └── qwen2_cram_patch.py
├── nq_100_qwen05.json              # head-selection input
└── transformer-explainer/          # Qwen visualization and live audience demo
    ├── trace_backend/
    └── src/
```

## 1. What was implemented

The original Transformer Explainer ran GPT-2 in the browser. This version uses
`Qwen/Qwen1.5-0.5B-Chat` in a Python backend and sends real model activations to
the Svelte visualization.

The project now includes:

- real Qwen tokenization and token IDs;
- all 24 Qwen decoder blocks and 16 attention heads;
- RMSNorm, separate Q/K/V projections, RoPE, causal attention, and SwiGLU;
- captured embeddings, Q/K/V vectors, attention scores, masks, softmax values,
  attention outputs, MLP values, final normalization, logits, and probabilities;
- word-level labels that retain the underlying Qwen token ranges;
- side-by-side vanilla and CrAM traces;
- a credibility control for selecting and down-weighting token spans;
- the CrAM formula shown at the attention stage;
- a live presenter, QR-code audience page, question collection, batch evaluation,
  and links from results into the explainer;
- responsive scaling for normal and long inspection traces;
- Qwen-specific influential-head selection generated from 100 NQ examples.

### How CrAM is applied

The original implementation is in:

```text
CrAM/utils/re_weighting.py
```

CrAM assigns credibility weights to passage tokens and adds their logarithms to
the attention logits before softmax:

```text
A_h^CrAM = Norm_1(A_h * s_bar)
         = softmax(Z_h + log(s_bar))    when s_bar > 0
```

A credibility of zero becomes a negative-infinity attention bias and therefore
suppresses the corresponding key columns.

The trace backend directly imports and registers the official
`Re_Weighting_Strategy.edit_attention_mask` method from
`CrAM/utils/re_weighting.py`. A second read-only hook captures the edited mask
after the official hook, and the displayed attention probabilities come from
Qwen's `output_attentions` result. The browser renders those captured values; it
does not implement another CrAM calculation.

The backend loads the generated influential-head mapping from:

```text
CrAM/datasets/nq_0.5b/qwen/selected_heads.json
```

The current mapping contains 64 selected heads from the model's 384 total heads.
Both interactive traces and audience batch evaluation apply CrAM only to those
heads. Non-selected heads remain vanilla and are identified as such in the UI.

## 2. Prerequisites

Install:

- Python 3.9;
- Node.js 20 or newer;
- npm 10 or newer;
- Git;
- `jq` for the optional JSON validation commands.

The tested dependency combination uses PyTorch 2.3.0 and Transformers 4.40.2.
Avoid casually upgrading Transformers: the Qwen eager-attention signature and
mask behavior are version-sensitive.

Model weights are downloaded from Hugging Face on the first backend run. Internet
access is therefore required once unless the model is already cached.

## 3. Installation from scratch

Clone the repository, then enter its root directory:

```bash
git clone <repository-url> CrAM
cd CrAM
```

Create and activate a Python 3.9 virtual environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install the unified CrAM and explainer backend from the single parent
requirements file:

```bash
python -m pip install -r requirements.txt
```

The public demo does not require FAISS, LangChain, an OpenAI client, or a live
retrieval pipeline.

Install the frontend:

```bash
cd transformer-explainer
npm ci
```

If `package-lock.json` is intentionally being regenerated, use `npm install`
instead.

## 4. Running Qwen + CrAM

Use two terminals.

### Terminal 1: Qwen backend, presenter, and audience

```bash
cd /path/to/CrAM/transformer-explainer
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  ../.venv/bin/python trace_backend/server.py --host 0.0.0.0 --port 8200
```

The startup message should report:

```text
CrAM uses 64 selected heads from .../datasets/nq_0.5b/qwen/selected_heads.json
```

Device selection defaults to CPU in the trace server. To request Apple MPS:

```bash
CRAM_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1 \
  ../.venv/bin/python trace_backend/server.py --host 0.0.0.0 --port 8200
```

To explicitly use CPU:

```bash
CRAM_DEVICE=cpu \
  ../.venv/bin/python trace_backend/server.py --host 0.0.0.0 --port 8200
```

### Terminal 2: explainer frontend

```bash
cd /path/to/CrAM/transformer-explainer
npm run dev -- --host 0.0.0.0
```

Open:

- presenter and question workflow: <http://localhost:8200/>;
- phone/audience page: <http://localhost:8200/audience>;
- Qwen + CrAM explainer: <http://localhost:5173/>.

Do not open `presenter.html` with a `file://` URL. It must be served by the
backend so its API requests use port 8200.

Phones must be on the same network and must use the LAN URL displayed beside the
QR code. The firewall must allow incoming connections to port 8200.

## 5. Stopping and restarting

Normally press `Ctrl+C` in both terminals.

If the terminals were lost, stop processes by port:

```bash
lsof -ti tcp:5173 | xargs kill
lsof -ti tcp:8200 | xargs kill
```

Then run the two commands from section 4 again.

## 6. Influential-head generation for Qwen1.5-0.5B

Never reuse head scores from another model size. Head indices are specific to a
model architecture and checkpoint.

The repository includes `nq_100_qwen05.json`, a prepared 100-example input with
one false passage and four retrieved passages per example. To use another
dataset, prepare the same fields: `question`, `wrong answer`, `ori_fake`, and
`reranked_dense_ctxs`.

Generate the per-example, per-head indirect-effect scores:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -c "
from utils.find_best_heads import casual_tracing_per_head

casual_tracing_per_head(
    filepath='nq_100_qwen05.json',
    LLM='Qwen/Qwen1.5-0.5B-Chat',
    output_dir='datasets/nq_0.5b',
    contexts_type=['ori_fake', 'reranked_dense'],
    topk=4,
)
"
```

The function name is spelled `casual_tracing_per_head` in the existing source.
The script saves after each example and resumes from `heads_scores.json` when
rerun.

Aggregate and sort all 384 heads:

```bash
.venv/bin/python -c "
from utils.find_best_heads import casual_tracing_combine_all
casual_tracing_combine_all('datasets/nq_0.5b/qwen')
"
```

Select the top 64:

```bash
.venv/bin/python -c "
from utils.find_best_heads import find_top_k_heads
find_top_k_heads('datasets/nq_0.5b/qwen', topk=64)
"
```

Validate:

```bash
jq 'length' datasets/nq_0.5b/qwen/heads_scores.json
jq 'length' datasets/nq_0.5b/qwen/heads_scores_mean.json
jq '[.[] | length] | add' datasets/nq_0.5b/qwen/selected_heads.json
```

Expected values:

```text
100
384
64
```

To use a mapping stored elsewhere:

```bash
CRAM_SELECTED_HEADS=/absolute/path/to/selected_heads.json \
  ../.venv/bin/python trace_backend/server.py --host 0.0.0.0 --port 8200
```

## 7. Running another Qwen checkpoint

For another Qwen checkpoint:

1. Change `MODEL_NAME` in `trace_backend/capture.py`.
2. Generate new `heads_scores.json`, `heads_scores_mean.json`, and
   `selected_heads.json` using that exact checkpoint.
3. Start the backend with `CRAM_SELECTED_HEADS` pointing to the new mapping.
4. Update `modelMetaMap.qwen` in `src/store/index.ts` if the layer count, head
   count, or hidden size changed.
5. Confirm the model's module names still include:
   `model.layers`, `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `input_layernorm`, `post_attention_layernorm`, and the SwiGLU projections.
6. Confirm eager attention accepts the per-head additive mask. Qwen2 models on
   Transformers 4.40.2 use `utils/qwen2_cram_patch.py`.
7. Run vanilla and CrAM on a short known example and check that selected heads
   change while non-selected heads remain identical.

### Important: grouped-query attention

The present capturer assumes multi-head attention where the number of key/value
heads equals the number of query heads. Qwen checkpoints using grouped-query
attention have fewer K/V heads. For those models, update `capture.py` to:

- reshape Q using `num_attention_heads`;
- reshape K and V using `num_key_value_heads`;
- repeat K/V groups to query-head count before score and value multiplication;
- ensure the selected-head indices refer to query heads.

Without this adjustment, a grouped-query Qwen checkpoint will produce invalid
reshapes or misleading head visualizations.

## 8. Running a different model family

Porting to Llama, Mistral, Gemma, or another architecture requires more than
changing the model name:

1. Identify the decoder-layer, attention, normalization, and MLP module paths.
2. Replace the Qwen-specific forward hooks in `trace_backend/capture.py`.
3. Implement the architecture's positional encoding and K/V-head behavior.
4. Verify where its attention mask is added relative to softmax.
5. Adapt or remove `utils/qwen2_cram_patch.py`.
6. Regenerate influential-head scores for that exact checkpoint.
7. Update frontend model metadata and terminology.
8. Validate captured tensors numerically against the model's returned
   attentions.

CrAM itself remains the same conceptual operation: add token log-credibility to
the selected heads' pre-softmax attention logits. The integration layer changes
because model implementations expose attention differently.

## 9. Verification and troubleshooting

Check that the backend is alive:

```bash
curl http://localhost:8200/api/config
```

The response should include:

```text
cram_selected_head_count: 64
```

Check the frontend build:

```bash
cd /path/to/CrAM/transformer-explainer
npm run build
```

### `selected_heads.json` missing

Generate it using section 6 or provide an override:

```bash
export CRAM_SELECTED_HEADS=/absolute/path/to/selected_heads.json
```

The backend intentionally fails at startup rather than silently modifying every
head.

### Backend connection failure

Confirm port 8200 is listening:

```bash
lsof -nP -iTCP:8200 -sTCP:LISTEN
```

The frontend currently calls `http://127.0.0.1:8200`, so the explainer browser
must run on the backend computer. Audience phones only need access to port 8200.

### Address already in use

```bash
lsof -ti tcp:8200 | xargs kill
lsof -ti tcp:5173 | xargs kill
```

### Model is slow

- The first run downloads and loads the checkpoint.
- CPU inference is reliable but slower.
- Apple MPS can be requested with `CRAM_DEVICE=mps`.
- Influential-head generation performs one baseline plus one forward pass for
  every head per example and therefore takes much longer than normal inference.

### Dependency mismatch

Recreate the environment instead of upgrading packages individually:

```bash
cd /path/to/CrAM
deactivate 2>/dev/null || true
rm -rf .venv
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Only remove `.venv` when it is safe to discard the existing environment.
