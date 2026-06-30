from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import asyncio
import json
import threading
from typing import List, Optional

import audience as aud
from capture import MODEL_NAME, TraceCapturer
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC = os.path.join(HERE, "static")

app = FastAPI(title="Qwen Transformer Explainer — Trace Server")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

session = aud.AudienceSession()
_capturer = None

import torch

_model_lock = threading.Lock()  # serialize all model use (batch + /api/trace)
ANS_MAX_TOKENS = 24  # short answers — bounded generation time


def capturer():
    global _capturer
    if _capturer is None:
        # CPU by default: clean fp32 numerics for the visualization, and Qwen-0.5B
        # is fast enough on CPU for short prompts. Override with CRAM_DEVICE=mps.
        _capturer = TraceCapturer(device=os.environ.get("CRAM_DEVICE", "cpu"))
    return _capturer


class TraceRequest(BaseModel):
    text: str = "The capital of France is"
    span: Optional[List[float]] = None  # [start, end] token indices to down-weight
    credibility: float = 0.0
    max_tokens: int = 16
    compact: bool = True


def _conflict_prompt(question: str, truth: str, claim: str):
    """Build the exact batch prompt and locate both passage token spans."""
    from utils.prompt import get_prompt

    cap = capturer()
    tok = cap.tokenizer
    content = get_prompt(
        context=[truth, claim], question=question, answer="", type="with_contexts"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    prompt = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    enc = tok(prompt, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()

    def token_span(text: str, fallback: str):
        target = text
        start = prompt.find(target)
        if start < 0:
            target = fallback
            start = prompt.find(target)
        if start < 0:
            return None
        end = start + len(target)
        ids = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
        return (ids[0], ids[-1] + 1) if ids else None

    return (
        prompt,
        enc,
        token_span(f"Passage-0: {truth}", truth),
        token_span(f"Passage-1: {claim}", claim),
    )


@app.on_event("startup")
def _warm():
    capturer()  # load the model at boot so the first request is fast
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        os.environ.setdefault("CRAM_LAN_IP", s.getsockname()[0])
    except OSError:
        pass
    finally:
        s.close()
    os.environ.setdefault("CRAM_PORT", os.environ.get("CRAM_PORT", "8200"))
    port = os.environ["CRAM_PORT"]
    print(
        f"[trace-server] loaded {MODEL_NAME} on {_capturer.device}; "
        f"CrAM uses {_capturer.cfg['cram_selected_head_count']} selected heads from "
        f"{_capturer.selected_heads_path}",
        flush=True,
    )
    print(
        f"[audience] presenter http://localhost:{port}/   "
        f"phone http://{os.environ.get('CRAM_LAN_IP')}:{port}/audience",
        flush=True,
    )


@app.get("/api/config")
def config():
    return capturer().cfg


@app.post("/api/trace")
def trace(req: TraceRequest):
    cap = capturer()
    with _model_lock:
        vanilla = cap.capture(req.text, max_tokens=req.max_tokens, compact=req.compact)
        cram = None
        if req.span and len(req.span) == 2:
            cram = cap.capture(
                req.text,
                max_tokens=req.max_tokens,
                compact=req.compact,
                cram={
                    "span": (int(req.span[0]), int(req.span[1])),
                    "credibility": req.credibility,
                },
            )
    return {"vanilla": vanilla, "cram": cram}


def _lan_audience_url() -> str:
    # A public tunnel URL (e.g. Cloudflare Tunnel) overrides the LAN IP so phones
    # on isolated networks (university/conference Wi-Fi) can still reach the page.
    pub = os.environ.get("CRAM_PUBLIC_URL")
    if pub:
        return pub.rstrip("/") + "/audience"
    ip = os.environ.get("CRAM_LAN_IP", "127.0.0.1")
    port = os.environ.get("CRAM_PORT", "8200")
    return f"http://{ip}:{port}/audience"


_NOCACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


@app.get("/", response_class=HTMLResponse)
def presenter_page():
    return FileResponse(os.path.join(STATIC, "presenter.html"), headers=_NOCACHE)


@app.get("/audience", response_class=HTMLResponse)
def audience_page():
    return FileResponse(os.path.join(STATIC, "audience.html"), headers=_NOCACHE)


@app.get("/qr")
def qr_png():
    url = _lan_audience_url()
    try:
        from io import BytesIO

        import qrcode
        from fastapi.responses import Response

        buf = BytesIO()
        qrcode.make(url).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except ImportError:
        return {"audience_url": url}


@app.get("/api/audience/questions")
def aud_questions():
    return {
        "phase": session.phase,
        "audience_url": _lan_audience_url(),
        "max_tokens": aud.MAX_CLAIM_TOKENS,
        "questions": [
            {"id": q["id"], "title": q["title"], "question": q["question"]}
            for q in session.questions.values()
        ],
    }


class SubmitReq(BaseModel):
    question_id: int
    text: str


@app.post("/api/audience/submit")
def aud_submit(req: SubmitReq, request: Request):
    ntok = len(capturer().tokenizer(req.text)["input_ids"])
    if ntok > aud.MAX_CLAIM_TOKENS:
        return {
            "ok": False,
            "error": f"Too long: {ntok} tokens (max {aud.MAX_CLAIM_TOKENS}).",
        }
    ip = request.client.host if request.client else ""
    return session.submit(req.question_id, req.text, ip)


@app.get("/api/audience/scoreboard")
def aud_scoreboard():
    return session.scoreboard()


@app.get("/api/audience/events")
async def aud_events():
    async def gen():
        last = -1
        while True:
            if session.rev != last:
                last = session.rev
                yield f"data: {json.dumps(session.scoreboard())}\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/audience/reset")
def aud_reset():
    session.reset()
    return session.scoreboard()


@app.post("/api/audience/end-vote")
def aud_end_vote():
    if session.phase == "collecting":
        session.phase = "processing"
        session.rev += 1
        threading.Thread(target=run_batch, daemon=True).start()
    return session.scoreboard()


@app.get("/api/audience/results/{qid}")
def aud_results(qid: int):
    q = session.questions.get(qid)
    return {
        "phase": session.phase,
        "progress": session.progress,
        "question": None
        if not q
        else {
            k: q[k] for k in ("id", "title", "question", "correct", "wrong", "truth")
        },
        "results": session.results.get(qid),
    }


@app.get("/api/audience/inspect/{qid}/{claim_id}")
def aud_inspect(qid: int, claim_id: str):
    """Return a focused Vanilla/CrAM trace for one completed demo claim."""
    q = session.questions.get(qid)
    result = session.results.get(qid) or {}
    claim = next((c for c in result.get("claims", []) if c["id"] == claim_id), None)
    if not q or not claim:
        raise HTTPException(status_code=404, detail="Claim result not found")

    prompt, enc, truth_span, claim_span = _conflict_prompt(
        q["question"], q["truth"], claim["text"]
    )
    if claim_span is None:
        raise HTTPException(
            status_code=422, detail="Could not locate the fake passage in the prompt"
        )

    # Keep the response/browser manageable while inference still uses the full
    # prompt. Show the fake passage, some truthful context, and the answer slot.
    total = enc["input_ids"].shape[1]
    claim_ids = list(range(*claim_span))
    truth_ids = list(range(*truth_span)) if truth_span else []
    budget = 24
    kept_claim = claim_ids[: min(len(claim_ids), 16)]
    remaining = max(0, budget - len(kept_claim) - 1)
    shown = truth_ids[-remaining:] + kept_claim + [total - 1]
    shown = sorted(set(shown))
    shown_claim = [shown.index(i) for i in kept_claim if i in shown]
    display_span = [min(shown_claim), max(shown_claim) + 1]
    max_tokens = total

    with _model_lock:
        vanilla = capturer().capture(
            prompt, max_tokens=max_tokens, compact=True, display_indices=shown
        )
        cram = capturer().capture(
            prompt,
            max_tokens=max_tokens,
            compact=True,
            display_indices=shown,
            cram={"span": claim_span, "credibility": 0.0},
        )
    return {
        "vanilla": vanilla,
        "cram": cram,
        "inspection": {
            "question_id": qid,
            "claim_id": claim_id,
            "question": q["question"],
            "truth": q["truth"],
            "claim": claim["text"],
            "source": claim["source"],
            "vanilla_answer": claim["vanilla"],
            "cram_answer": claim["cram"],
            "display_span": display_span,
            "full_claim_span": list(claim_span),
        },
    }


@torch.no_grad()
def _answer(question: str, truth: str, claim: str, suppress_claim: bool) -> str:
    """Short greedy answer for the conflict prompt. If suppress_claim, the claim's
    attention is fully down-weighted on the generated influential-head mapping
    (CrAM). Hooks are ALWAYS removed and all model use is under _model_lock."""
    cap = capturer()
    tok, model, device = cap.tokenizer, cap.model, cap.device
    prompt, enc, _, claim_span = _conflict_prompt(question, truth, claim)
    input_ids = enc["input_ids"].to(device)
    T = input_ids.shape[1]

    span = claim_span if suppress_claim else None

    with _model_lock:
        handles = []
        if span:
            s, e = span
            credibility = torch.ones(
                (1, T), dtype=torch.float32, device=device
            )
            credibility[:, s:e] = 0.0
            official_attention_weight = torch.log(credibility)

            for layer_idx, heads in cap.selected_heads.items():
                layer = model.model.layers[layer_idx]
                handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        cap.official_cram_hook(official_attention_weight, heads),
                        with_kwargs=True,
                    )
                )
        try:
            out = model.generate(
                input_ids,
                max_new_tokens=ANS_MAX_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.eos_token_id,
            )
        finally:
            for h in handles:
                h.remove()
    return (
        tok.decode(out[0][T:], skip_special_tokens=True).strip().replace("\n", " ")[:90]
    )


def _judge(ans: str, refs: List[str], wrong: str):
    a = ans.lower()
    return any(r.lower() in a for r in refs), (bool(wrong) and wrong.lower() in a)


def run_batch():
    rid = session.round_id  # if a reset bumps this, stop writing
    try:
        plans = {qid: session.sampled_pool(qid) for qid in session.questions}
        total = sum(len(v) for v in plans.values())
        aud_n = sum(1 for v in plans.values() for c in v if c["source"] == "audience")
        session.progress = {
            "done": 0,
            "total": total,
            "stage": "running",
            "audience": aud_n,
            "dataset": total - aud_n,
        }
        session.rev += 1
        for qid, claims in plans.items():
            if session.round_id != rid:
                return
            q = session.questions[qid]
            out = []
            for c in claims:
                if session.round_id != rid:
                    return
                try:
                    van = _answer(
                        q["question"], q["truth"], c["text"], suppress_claim=False
                    )
                    cram = _answer(
                        q["question"], q["truth"], c["text"], suppress_claim=True
                    )
                    vc, _ = _judge(van, q["correct"], q["wrong"])
                    cc, _ = _judge(cram, q["correct"], q["wrong"])
                    out.append(
                        {
                            "id": c["id"],
                            "text": c["text"],
                            "source": c["source"],
                            "vanilla": van,
                            "cram": cram,
                            "vanilla_correct": vc,
                            "cram_correct": cc,
                            "flip": (not vc) and cc,
                        }
                    )
                except Exception as exc:
                    print("[audience] claim failed:", repr(exc), flush=True)
                    out.append(
                        {
                            "id": c["id"],
                            "text": c["text"],
                            "source": c["source"],
                            "vanilla": "(error)",
                            "cram": "(error)",
                            "vanilla_correct": False,
                            "cram_correct": False,
                            "flip": False,
                        }
                    )
                # store partial results after every claim so the UI can render progress
                session.results[qid] = {
                    "n": len(out),
                    "vanilla_fooled": sum(1 for r in out if not r["vanilla_correct"]),
                    "cram_defended": sum(1 for r in out if r["cram_correct"]),
                    "flips": sum(1 for r in out if r["flip"]),
                    "claims": out,
                }
                session.progress["done"] += 1
                session.rev += 1
        if session.round_id == rid:
            session.phase = "done"
    except Exception as exc:
        print("[audience] batch failed:", repr(exc), flush=True)
        session.progress["error"] = str(exc)[:200]
        session.phase = "done"
    finally:
        session.rev += 1


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="0.0.0.0")  # LAN-reachable for phones
    args = ap.parse_args()
    os.environ["CRAM_PORT"] = str(args.port)
    uvicorn.run(app, host=args.host, port=args.port)
