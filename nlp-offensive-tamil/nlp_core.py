"""nlp_core.py - shared modules for the Tamil-English offensive language detector.

Modules: preprocessing, script identification, input guard, transformer predictor
(sliding window + abstention) and classical baseline predictor. Used by the notebook,
the tests and the Gradio app so that every entry point runs the same code path.
"""
import json
import os
import re
import time
import unicodedata

import numpy as np

LABELS = ["Not_offensive", "Offensive_Untargeted", "Offensive_Targeted_Individual",
          "Offensive_Targeted_Group", "Offensive_Targeted_Other", "Not_Tamil"]
PRETTY = {
    "Not_offensive": "Not offensive",
    "Offensive_Untargeted": "Offensive (untargeted)",
    "Offensive_Targeted_Individual": "Offensive - targets an individual",
    "Offensive_Targeted_Group": "Offensive - targets a group",
    "Offensive_Targeted_Other": "Offensive - targets something else",
    "Not_Tamil": "Not Tamil (out of scope)",
}
OFFENSIVE_IDS = [1, 2, 3, 4]
MAX_CHARS = 20000          # hard cap to protect latency / memory

# ------------------------------------------------------------------ preprocessing
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_ELONG = re.compile(r"([^\d\s])\1{2,}")   # keeps numbers like 1000 intact
_WS = re.compile(r"\s+")


def clean_text(text, lower=False):
    """Normalise a raw comment. Safe on None / non-string input."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _URL.sub(" URL ", text)
    text = _MENTION.sub(" @USER ", text)
    text = _HASHTAG.sub(r" \1 ", text)
    text = _ELONG.sub(r"\1\1", text)          # sooooo -> soo, !!!!! -> !!
    text = _WS.sub(" ", text).strip()
    return text.lower() if lower else text


def clean_lower(text):
    """Module-level (picklable) preprocessor for the TF-IDF baseline."""
    return clean_text(text, lower=True)


def _is_tamil(ch):
    return "\u0B80" <= ch <= "\u0BFF"


def script_profile(text):
    """Fraction of Tamil-script, Latin and other-script letters, plus a coarse type."""
    text = "" if text is None else str(text)
    ta = sum(_is_tamil(c) for c in text)
    la = sum(c.isascii() and c.isalpha() for c in text)
    ot = sum(c.isalpha() and not c.isascii() and not _is_tamil(c) for c in text)
    tot = ta + la + ot
    if tot == 0:
        return {"tamil": 0.0, "latin": 0.0, "other": 0.0, "type": "no letters"}
    r = {"tamil": ta / tot, "latin": la / tot, "other": ot / tot}
    if r["other"] >= 0.5:
        kind = "other script"
    elif r["tamil"] >= 0.8:
        kind = "Tamil script"
    elif r["latin"] >= 0.8:
        kind = "romanized"
    else:
        kind = "mixed script"
    r = {k: round(v, 3) for k, v in r.items()}
    r["type"] = kind
    return r


# ------------------------------------------------------------------ shared inference helpers
def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _base_result(text):
    return {"input_chars": 0 if text is None else len(str(text)), "status": "ok", "label": None,
            "pretty_label": None, "confidence": None, "probs": {}, "warnings": [], "n_chunks": 0,
            "aggregation": None, "script": None, "cleaned": "", "n_tokens": None,
            "unk_ratio": None, "tokens_per_word": None, "latency_ms": None}


def _guard(text, r):
    """Input guard. Returns cleaned text, or None when there is nothing to classify."""
    if text is None or not str(text).strip():
        r["status"] = "empty_input"
        r["warnings"].append("Input is empty - nothing to classify.")
        return None
    text = str(text)
    if len(text) > MAX_CHARS:
        r["warnings"].append(f"Input truncated from {len(text):,} to {MAX_CHARS:,} characters.")
        text = text[:MAX_CHARS]
    cleaned = clean_text(text)
    r["cleaned"] = cleaned
    r["script"] = script_profile(cleaned)
    if not any(c.isalnum() for c in cleaned):
        r["warnings"].append("No letters or digits (emoji/punctuation only) - low-evidence prediction.")
    return cleaned


def _finalise(probs, r, tau, t0):
    pid = int(np.argmax(probs))
    conf = float(probs[pid])
    r["label"] = LABELS[pid]
    r["pretty_label"] = PRETTY[LABELS[pid]]
    r["confidence"] = round(conf, 4)
    r["probs"] = {LABELS[i]: float(p) for i, p in enumerate(probs)}
    if r["script"] and r["script"]["type"] == "other script":
        r["status"] = "out_of_scope_language"
        r["warnings"].append("Mostly non-Tamil, non-Latin script - outside the training distribution.")
    elif conf < tau:
        r["status"] = "uncertain"
        r["warnings"].append(f"Confidence {conf:.2f} < threshold tau={tau:.2f}: abstain and route to a human.")
    r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return r


def save_config(art_dir, cfg):
    with open(os.path.join(art_dir, "inference_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def load_config(art_dir):
    with open(os.path.join(art_dir, "inference_config.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------ transformer predictor
class TransformerPredictor:
    """Fine-tuned encoder + sliding window for long input + confidence abstention."""

    def __init__(self, model_dir=None, device=None, max_len=128, stride=32, tau=0.0,
                 model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_dir)
        m = model if model is not None else AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model = m.to(self.device).eval()
        self.max_len, self.stride, self.tau = int(max_len), int(stride), float(tau)
        self.name = "MuRIL transformer"

    @classmethod
    def from_artifacts(cls, art_dir, device=None):
        cfg = load_config(art_dir)
        return cls(os.path.join(art_dir, "transformer"), device=device, max_len=cfg["max_len"],
                   stride=cfg["stride"], tau=cfg["tau"])

    def chunk_probs(self, cleaned):
        enc = self.tok(cleaned, truncation=True, max_length=self.max_len, stride=self.stride,
                       return_overflowing_tokens=True, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask", "token_type_ids")}
        with self.torch.inference_mode():
            logits = self.model(**enc).logits.float().cpu().numpy()
        return softmax(logits)

    def predict(self, text):
        t0 = time.perf_counter()
        r = _base_result(text)
        cleaned = _guard(text, r)
        if cleaned is None:
            r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return r
        ids = self.tok(cleaned, add_special_tokens=False)["input_ids"]
        unk = self.tok.unk_token_id
        r["n_tokens"] = len(ids)
        r["unk_ratio"] = round(sum(i == unk for i in ids) / max(len(ids), 1), 3)
        r["tokens_per_word"] = round(len(ids) / max(len(cleaned.split()), 1), 2)
        if r["unk_ratio"] > 0.2:
            r["warnings"].append(f"{r['unk_ratio']:.0%} of tokens are unknown to the vocabulary.")
        P = self.chunk_probs(cleaned)
        r["n_chunks"] = len(P)
        off = P[:, OFFENSIVE_IDS].sum(1)
        if len(P) > 1 and off.max() > 0.5:
            probs, r["aggregation"] = P[int(off.argmax())], "max-offensive chunk"
        else:
            probs = P.mean(0)
            r["aggregation"] = "mean over chunks" if len(P) > 1 else "single window"
        if len(P) > 1:
            r["warnings"].append(f"Longer than the {self.max_len}-token window: split into {len(P)} "
                                 f"overlapping chunks ({r['aggregation']}).")
        return _finalise(probs, r, self.tau, t0)


# ------------------------------------------------------------------ classical baseline predictor
class BaselinePredictor:
    """TF-IDF + linear classifier behind the same interface as the transformer."""

    def __init__(self, pipeline, tau=0.0):
        self.pipe, self.tau = pipeline, float(tau)
        self.name = "TF-IDF baseline"

    @classmethod
    def from_artifacts(cls, art_dir):
        import joblib
        cfg = load_config(art_dir)
        return cls(joblib.load(os.path.join(art_dir, "baseline.joblib")), tau=cfg.get("baseline_tau", 0.0))

    def predict_proba(self, texts):
        if hasattr(self.pipe, "predict_proba"):
            return self.pipe.predict_proba(texts)
        return softmax(self.pipe.decision_function(texts))

    def predict(self, text):
        t0 = time.perf_counter()
        r = _base_result(text)
        cleaned = _guard(text, r)
        if cleaned is None:
            r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return r
        probs = self.predict_proba([cleaned])[0]
        r["n_chunks"], r["aggregation"] = 1, "bag of n-grams (no length limit)"
        return _finalise(probs, r, self.tau, t0)