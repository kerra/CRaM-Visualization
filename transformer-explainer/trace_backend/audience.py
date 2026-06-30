from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(REPO_ROOT, "curated_short_conflict_trivia.json")

# The three demo questions (validated clean vanilla->CrAM flips).
DEMO_IDS = [8174, 1849, 20003]

# sampling targets
TARGET_X = 8  # max statements per question to actually run
MIN_X = 3  # below this, top up from the dataset's ori_fake pool
MAX_CLAIM_TOKENS = 15


def _short_title(question: str) -> str:
    # full question text (the UI wraps it); kept as a field for the frontends
    return question.strip().strip('"')


def load_questions() -> Dict[int, dict]:
    data = json.load(open(DATASET, encoding="utf-8"))
    by_id = {d["id"]: d for d in data}
    out = {}
    for qid in DEMO_IDS:
        d = by_id[qid]
        truth = (d.get("reranked_dense_ctxs") or d.get("dense_ctxs") or [""])[0]
        out[qid] = {
            "id": qid,
            "question": d["question"].strip().strip('"'),
            "title": _short_title(d["question"]),
            "truth": truth,  # short curated truth passage
            "correct": d["reference"],  # list of acceptable answers
            "wrong": d.get("wrong answer", ""),
            "fake_pool": list(d.get("ori_fake", [])),  # for topping up sparse pools
        }
    return out


@dataclass
class ClaimResult:
    claim_id: str
    text: str
    source: str  # "audience" | "dataset"
    vanilla: str = ""
    cram: str = ""
    vanilla_correct: bool = False
    cram_correct: bool = False
    vanilla_wrong: bool = False
    flip: bool = False  # vanilla wrong -> CrAM correct


@dataclass
class AudienceSession:
    questions: Dict[int, dict] = field(default_factory=load_questions)
    phase: str = "collecting"  # collecting | processing | done
    pools: Dict[int, List[dict]] = field(default_factory=dict)  # qid -> [{id,text,ip}]
    results: Dict[int, dict] = field(default_factory=dict)  # qid -> processed results
    progress: dict = field(default_factory=lambda: {"done": 0, "total": 0})
    _ip_counts: Dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    rev: int = 0  # bumps on any change (for SSE / polling)
    round_id: int = 0  # bumps on reset; a stale batch thread checks this and stops

    def __post_init__(self):
        for qid in self.questions:
            self.pools.setdefault(qid, [])

    # ----------------------------------------------------------- collection
    def reset(self):
        with self.lock:
            self.phase = "collecting"
            self.pools = {qid: [] for qid in self.questions}
            self.results = {}
            self.progress = {"done": 0, "total": 0}
            self._ip_counts = {}
            self.round_id += 1  # invalidate any in-flight batch thread
            self.rev += 1

    def scoreboard(self) -> dict:
        return {
            "phase": self.phase,
            "rev": self.rev,
            "counts": {str(qid): len(self.pools[qid]) for qid in self.questions},
            "total": sum(len(v) for v in self.pools.values()),
            "progress": self.progress,
        }

    def submit(self, question_id: int, text: str, ip: str = "") -> dict:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if self.phase != "collecting":
            return {"ok": False, "error": "Voting has ended."}
        if question_id not in self.questions:
            return {"ok": False, "error": "Unknown question."}
        if len(text) < 4:
            return {"ok": False, "error": "Statement too short."}
        if self._ip_counts.get(ip, 0) >= 5:
            return {"ok": False, "error": "Submission limit reached."}
        with self.lock:
            self.pools[question_id].append(
                {"id": uuid.uuid4().hex[:8], "text": text, "ip": ip}
            )
            self._ip_counts[ip] = self._ip_counts.get(ip, 0) + 1
            self.rev += 1
        return {"ok": True}

    # ----------------------------------------------------------- sampling
    def sampled_pool(self, qid: int, seed: int = 42) -> List[dict]:
        """Audience statements for a question, sampled/topped-up to a runnable set."""
        rng = random.Random(seed)
        pool = [
            {"id": p["id"], "text": p["text"], "source": "audience"}
            for p in self.pools[qid]
        ]
        if len(pool) > TARGET_X:
            pool = rng.sample(pool, TARGET_X)
        if len(pool) < MIN_X:
            fakes = self.questions[qid]["fake_pool"]
            need = MIN_X - len(pool)
            for f in fakes[:need]:
                pool.append(
                    {"id": "ds_" + uuid.uuid4().hex[:6], "text": f, "source": "dataset"}
                )
        return pool
