# -*- coding: utf-8 -*-
"""
Seq2Seq Reverse Attribution (Claims) with Captum IG/DeepLIFT - FIXED
"""

import os, re, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from captum.attr import IntegratedGradients, DeepLift

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception:
    nltk = None
    sent_tokenize = None

from helper import extract_text_from_pdf_robust, preprocess_pdf_text
from helper_addons import build_sentence_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def doi_candidates(doi: str) -> List[str]:
    d = doi.strip()
    variants = {d, d.replace("/", "_"), d.replace("/", "-"),
                d.replace(".", "_").replace("/", "_"),
                re.sub(r"[^0-9A-Za-z]+", "_", d),
                d.split("/")[-1]}
    return [v for v in variants if v]


def find_files_for_doi(doi: str, src_dir: Path) -> List[Path]:
    cands = doi_candidates(doi)
    found = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith((".pdf", ".txt")): continue
            if any(c in f for c in cands):
                found.append(Path(root) / f)
    seen, uniq = set(), []
    for p in found:
        s = str(p)
        if s not in seen:
            uniq.append(p); seen.add(s)
    return uniq


class Seq2SeqRevAttrIG:
    def __init__(self, model_name: str = "google/mt5-base", ig_steps: int = 32, use_deeplift: bool = False):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.ig_steps = ig_steps
        self.use_deeplift = use_deeplift

    def chunk_text(self, text: str, max_tokens: int = 768, stride: int = 128):
        t = self.tok(text, return_attention_mask=False, add_special_tokens=False)
        ids = t.input_ids
        chunks = []
        start = 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            chunk_ids = ids[start:end]
            chunk_text = self.tok.decode(chunk_ids, skip_special_tokens=True)
            chunks.append((chunk_text, start, end))
            if end == len(ids):
                break
            start += stride
        return chunks

    def _forward_core(self, inputs_embeds, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        """
        Forward function that returns per-sample negative loss.
        CRITICAL: Must return shape [batch_size] for Captum compatibility.
        """
        batch_size = inputs_embeds.shape[0]
        
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        
        # Return per-sample negative loss
        if batch_size == 1:
            return -out.loss.unsqueeze(0)
        else:
            # Compute per-sample loss manually
            logits = out.logits  # [batch_size, seq_len, vocab_size]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            # Reshape and average per sample
            losses = losses.view(batch_size, -1).mean(dim=1)  # [batch_size]
            return -losses

    def ig_token_attributions(self, chunk_text: str, gold_summary: str,
                              enc_max_len: int = 1024, lab_max_len: int = 512):
        enc = self.tok(
            chunk_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=enc_max_len,
        )
        lab = self.tok(
            gold_summary,
            return_tensors="pt",
            truncation=True,
            max_length=lab_max_len,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        lab = {k: v.to(DEVICE) for k, v in lab.items()}

        for p in self.model.parameters():
            p.requires_grad_(False)
        inputs_embeds = self.model.get_input_embeddings()(enc["input_ids"])
        inputs_embeds.requires_grad_(True)

        attention_mask = enc["attention_mask"]
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(lab["input_ids"])
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        decoder_attention_mask = (decoder_input_ids != pad_id).long().to(DEVICE)
        labels = lab["input_ids"]

        if self.use_deeplift:
            # DeepLift path - needs similar batch handling
            class _LossWrapper(nn.Module):
                def __init__(self, base_model, forward_core, attention_mask, decoder_input_ids, 
                           decoder_attention_mask, labels):
                    super().__init__()
                    self.base_model = base_model
                    self.forward_core = forward_core
                    self.attention_mask = attention_mask
                    self.decoder_input_ids = decoder_input_ids
                    self.decoder_attention_mask = decoder_attention_mask
                    self.labels = labels
                
                def forward(self, inputs_embeds):
                    current_batch_size = inputs_embeds.shape[0]
                    
                    # Expand fixed inputs to match batch size
                    expanded_attention_mask = self.attention_mask.expand(current_batch_size, -1).contiguous()
                    expanded_decoder_input_ids = self.decoder_input_ids.expand(current_batch_size, -1).contiguous()
                    expanded_decoder_attention_mask = self.decoder_attention_mask.expand(current_batch_size, -1).contiguous()
                    expanded_labels = self.labels.expand(current_batch_size, -1).contiguous()
                    
                    return self.forward_core(
                        inputs_embeds,
                        expanded_attention_mask,
                        expanded_decoder_input_ids,
                        expanded_decoder_attention_mask,
                        expanded_labels,
                    )
            
            wrapper = _LossWrapper(
                self.model,
                self._forward_core,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                labels,
            )
            explainer = DeepLift(wrapper)
            attrs = explainer.attribute(inputs=inputs_embeds, baselines=torch.zeros_like(inputs_embeds))
        else:
            # IntegratedGradients path - fixed for batching
            def forward_fn(inp):
                current_batch_size = inp.shape[0]
                
                # Expand fixed inputs to match batch size from integration steps
                expanded_attention_mask = attention_mask.expand(current_batch_size, -1).contiguous()
                expanded_decoder_input_ids = decoder_input_ids.expand(current_batch_size, -1).contiguous()
                expanded_decoder_attention_mask = decoder_attention_mask.expand(current_batch_size, -1).contiguous()
                expanded_labels = labels.expand(current_batch_size, -1).contiguous()
                
                return self._forward_core(
                    inp,
                    expanded_attention_mask,
                    expanded_decoder_input_ids,
                    expanded_decoder_attention_mask,
                    expanded_labels,
                )
            
            explainer = IntegratedGradients(forward_fn)
            attrs = explainer.attribute(
                inputs=inputs_embeds,
                baselines=torch.zeros_like(inputs_embeds),
                n_steps=self.ig_steps,
                internal_batch_size=1,
            )

        token_attr = attrs.norm(dim=-1).squeeze(0)
        token_attr = token_attr / (token_attr.sum() + 1e-12)
        tokens = self.tok.convert_ids_to_tokens(enc["input_ids"].detach().cpu().tolist()[0])
        offsets = enc["offset_mapping"].detach().cpu().tolist()[0]
        return tokens, offsets, token_attr.detach().cpu().numpy()

    def spans_from_token_scores(self, text: str, offsets, scores, top_k: int = 3,
                                min_span_len_chars: int = 25, smooth_window: int = 5, merge_gap: int = 20):
        s = np.array(scores, dtype=float)
        if smooth_window > 1:
            k = smooth_window
            s = np.convolve(s, np.ones(k)/k, mode="same")
        char_scores = np.zeros(len(text), dtype=float)
        for (a,b), sc in zip(offsets, s):
            if a < b and b <= len(text):
                char_scores[a:b] += float(sc)
        spans = []
        for _ in range(top_k * 3):
            idx = int(char_scores.argmax())
            if char_scores[idx] <= 0: break
            L = R = idx
            thresh = 0.25 * char_scores[idx]
            while L > 0 and char_scores[L-1] >= thresh: L -= 1
            while R+1 < len(char_scores) and char_scores[R+1] >= thresh: R += 1
            if R - L + 1 >= min_span_len_chars:
                spans.append((L, R+1, float(char_scores[L:R+1].sum())))
            char_scores[L:R+1] = 0.0
        spans.sort()
        merged = []
        for s0,e0,w0 in spans:
            if not merged: merged.append([s0,e0,w0]); continue
            s1,e1,w1 = merged[-1]
            if s0 - e1 <= merge_gap:
                merged[-1][1] = max(e1,e0); merged[-1][2] += w0
            else:
                merged.append([s0,e0,w0])
        merged.sort(key=lambda x: x[2], reverse=True)
        out = []
        for s0,e0,w in merged[:top_k]:
            out.append({"char_start": s0, "char_end": e0, "score": w, "snippet": text[s0:e0].strip()})
        return out

    def find_evidence_spans_for_summary(self, summary: str, docs: List[str],
                                        chunk_len: int = 768, stride: int = 128,
                                        spans_per_doc: int = 3):
        all_candidates = []
        for di, doc in enumerate(docs):
            chunks = self.chunk_text(doc, max_tokens=chunk_len, stride=stride)
            for (chunk_text, _s, _e) in chunks:
                tokens, offsets, scores = self.ig_token_attributions(chunk_text, summary)
                spans = self.spans_from_token_scores(chunk_text, offsets, scores, top_k=spans_per_doc)
                for sp in spans:
                    sp["doc_id"] = di
                    sp["chunk_text"] = chunk_text
                    all_candidates.append(sp)
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        return all_candidates


def split_summary_into_claims(summary: str) -> List[str]:
    if not summary: return []
    if sent_tokenize is None:
        parts = re.split(r'(?<=[.!?])\s+', summary.strip())
        return [p for p in parts if p]
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try: nltk.download("punkt")
        except Exception: pass
    sents = sent_tokenize(summary.strip())
    return [s for s in sents if s.strip()]


def map_spans_to_sentences(
    spans: List[Dict[str, Any]],
    doc_texts: Dict[str, str],
    sentence_maps: Dict[str, Dict[str, Any]],
    doc_ids: List[str],
    top_sentences_per_claim: int = 1,
) -> List[Dict[str, Any]]:
    """
    Map character-level spans (relative to a document) to the best-matching sentence(s).

    - Robust to interval payload shape: accepts dict payloads or Interval-like objects with .data
    - Deduplicates multiple spans that hit the same sentence, keeping the highest-scoring (and then longest-overlap) span
    - Returns up to `top_sentences_per_claim` global-best sentences (or all if <=0)

    Expected `spans` item keys:
      - "doc_id": int index into `doc_ids`
      - "char_start", "char_end": absolute char offsets within the document text
      - "score": span score (float)

    `sentence_maps[doc_id]` must contain:
      - "sid2span": Dict[int, Tuple[int,int]] mapping sentence id → (start,end) char offsets
      - "tree": an interval index with .search(s0, e0) -> iterable of payloads
               where each payload is either a dict with "sid" or an object with `.data` holding a dict or sid.
    """
    def _payload_to_sid(iv: Any) -> Optional[int]:
        """Normalize interval payloads to an integer sentence id."""
        # dict payload
        if isinstance(iv, dict):
            sid = iv.get("sid")
            try:
                return int(sid) if sid is not None else None
            except (TypeError, ValueError):
                return None
        # object with .data
        data = getattr(iv, "data", None)
        if data is not None:
            if isinstance(data, dict):
                try:
                    return int(data.get("sid")) if data.get("sid") is not None else None
                except (TypeError, ValueError):
                    return None
            try:
                return int(data)
            except (TypeError, ValueError):
                return None
        # bare int-like?
        try:
            return int(iv)
        except (TypeError, ValueError):
            return None

    hits_by_sentence: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for sp in spans or []:
        didx = sp.get("doc_id")
        if didx is None or not (0 <= int(didx) < len(doc_ids)):
            continue
        doc_id = doc_ids[int(didx)]

        text = doc_texts.get(doc_id)
        mapping = sentence_maps.get(doc_id) or {}
        if text is None or not mapping:
            continue

        tree = mapping.get("tree")
        sid2span = mapping.get("sid2span") or {}
        if tree is None or not sid2span:
            continue

        s0 = sp.get("char_start")
        e0 = sp.get("char_end")
        sc = float(sp.get("score", 0.0))

        # Basic sanity/bounds
        try:
            s0 = int(s0)
            e0 = int(e0)
        except (TypeError, ValueError):
            continue
        if s0 is None or e0 is None or s0 < 0 or e0 <= s0:
            continue
        doc_len = len(text)
        if s0 >= doc_len:
            continue
        s0 = max(0, s0)
        e0 = min(doc_len, e0)

        try:
            overlaps = tree.search(s0, e0)
        except Exception:
            overlaps = []

        for iv in overlaps:
            sid = _payload_to_sid(iv)
            if sid is None:
                continue
            st_en = sid2span.get(int(sid))
            if not st_en:
                continue
            st, en = st_en
            if st is None or en is None or not (0 <= st < en <= doc_len):
                continue

            # Overlap length for tie-breaks
            overlap_len = max(0, min(e0, en) - max(s0, st))
            if overlap_len == 0:
                # It’s possible the interval index returned an adjacent but non-overlapping sentence; skip.
                continue

            sent_text = text[st:en].strip()
            key = (doc_id, int(sid))

            prev = hits_by_sentence.get(key)
            if prev is None or (sc, overlap_len) > (prev["span_score"], prev["_overlap_len"]):
                hits_by_sentence[key] = {
                    "source_doc": doc_id,
                    "sentence_idx": int(sid),
                    "sentence": sent_text,
                    "doc_char_start": st,
                    "doc_char_end": en,
                    "span_char_start": s0,
                    "span_char_end": e0,
                    "span_score": float(sc),
                    "_overlap_len": int(overlap_len),  # internal tie-break helper
                }

    hits = list(hits_by_sentence.values())
    # Sort by best span score, then by overlap size, then by shorter sentence (optional)
    hits.sort(key=lambda x: (x["span_score"], x["_overlap_len"], -(x["doc_char_end"] - x["doc_char_start"])), reverse=True)

    # Clean internal field
    for h in hits:
        h.pop("_overlap_len", None)

    if top_sentences_per_claim and top_sentences_per_claim > 0:
        return hits[:top_sentences_per_claim]
    return hits

def retrieve_candidates_for_claim(
    claim: str,
    *,
    flexiconc_db: str,
    vector_backends: List[str],
    use_splade: bool,
    dense_weight: float,
    splade_weight: float,
    ce_model: str,
    ce_keep: int,
    per_doc_cap: int,
) -> List[Dict[str, Any]]:
    """
    Returns a list of {source_doc, sentence_idx, sentence, score_dense, score_splade, score_ce}
    from FlexiConc (dense + optional SPLADE) fused, then CrossEncoder-reranked.
    """
    try:
        import importlib
        fx = importlib.import_module("flexiconc_adapter")
    except Exception as e:
        print(f"⚠️ flexiconc_adapter not available: {e}")
        return []

    # ---- Stage 1: Dense retrieval (MPNet + BGE) via FAISS
    dense_hits: Dict[Tuple[str,int], Dict[str,Any]] = {}
    for vb in vector_backends:
        try:
            res = fx.query_concordance(
                flexiconc_db,
                [claim],                 # terms; adapter accepts list of strings
                mode="AND",
                limit=1000,              # gather a healthy pool
                vector_backend=vb,
                use_faiss=True,
            )
            rows = res.get("rows", []) if isinstance(res, dict) else (res or [])
        except Exception as e:
            print(f"⚠️ dense retrieval failed for {vb}: {e}")
            rows = []
        for r in rows:
            key = (r["doc_id"], int(r["sentence_idx"]))
            rec = dense_hits.setdefault(key, {
                "source_doc": r["doc_id"],
                "sentence_idx": int(r["sentence_idx"]),
                "sentence": r.get("text") or r.get("sentence") or "",
                "score_dense_accum": 0.0,
                "score_splade": 0.0,
            })
            # accumulate normalized score if provided; fall back to rank-based
            s = float(r.get("score", 0.0))
            rec["score_dense_accum"] += s

    # ---- Optional: SPLADE retrieval (hybrid)
    if use_splade:
        try:
            query_splade = getattr(fx, "query_splade")
        except Exception:
            query_splade = None
        if query_splade is None:
            print("ℹ️ SPLADE not available in adapter; skipping.")
        else:
            try:
                res_s = query_splade(flexiconc_db, claim, limit=1000)
                rows_s = res_s.get("rows", []) if isinstance(res_s, dict) else (res_s or [])
                for r in rows_s:
                    key = (r["doc_id"], int(r["sentence_idx"]))
                    rec = dense_hits.setdefault(key, {
                        "source_doc": r["doc_id"],
                        "sentence_idx": int(r["sentence_idx"]),
                        "sentence": r.get("text") or r.get("sentence") or "",
                        "score_dense_accum": 0.0,
                        "score_splade": 0.0,
                    })
                    rec["score_splade"] = max(rec["score_splade"], float(r.get("score", 0.0)))
            except Exception as e:
                print(f"⚠️ SPLADE retrieval failed: {e}")

    if not dense_hits:
        return []

    # ---- Simple fusion (z-norm each channel present)
    dense_scores = np.array([v["score_dense_accum"] for v in dense_hits.values()], float)
    dense_scores = (dense_scores - dense_scores.mean()) / (dense_scores.std() + 1e-8)
    if use_splade:
        splade_scores = np.array([v["score_splade"] for v in dense_hits.values()], float)
        splade_scores = (splade_scores - splade_scores.mean()) / (splade_scores.std() + 1e-8)
        fused = dense_weight * dense_scores + splade_weight * splade_scores
    else:
        fused = dense_scores

    keys = list(dense_hits.keys())
    for i, k in enumerate(keys):
        dense_hits[k]["score_fused"] = float(fused[i])

    # Keep a global pre-CE pool (cap generously)
    items = list(dense_hits.values())
    items.sort(key=lambda x: x["score_fused"], reverse=True)
    pre_ce = items[: max(2000, ce_keep * 5)]

    # ---- Stage 2: CrossEncoder reranking
    ce_scores = None
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(ce_model)
        pairs = [(claim, it["sentence"]) for it in pre_ce]
        ce_scores = ce.predict(pairs)
    except Exception as e:
        print(f"⚠️ CE unavailable or failed ({ce_model}): {e}")

    if ce_scores is not None:
        for it, sc in zip(pre_ce, ce_scores):
            it["score_ce"] = float(sc)
        pre_ce.sort(key=lambda x: x["score_ce"], reverse=True)
        post_ce = pre_ce[:ce_keep]
    else:
        # fallback to fused order
        for it in pre_ce:
            it["score_ce"] = 0.0
        post_ce = pre_ce[:ce_keep]

    # ---- Per-document cap before IG
    per_doc: Dict[str, List[Dict[str, Any]]] = {}
    for it in post_ce:
        per_doc.setdefault(it["source_doc"], []).append(it)
    final_pool = []
    for did, lst in per_doc.items():
        lst.sort(key=lambda x: (x["score_ce"], x["score_fused"]), reverse=True)
        final_pool.extend(lst[:per_doc_cap])

    return final_pool
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_csv", required=True)
    ap.add_argument("--row_iloc", type=int, default=0)
    ap.add_argument("--src_dir", default="/content/drive/MyDrive/ENLENS/summ_pdfs")
    ap.add_argument("--out_prefix", default="/content/revattr_row")
    ap.add_argument("--summary_col", default="Green Hydrogen Value Chain Justification_Justification")
    ap.add_argument("--dois_col", default="all_dois")
    ap.add_argument("--model_name", default="google/mt5-base")
    ap.add_argument("--chunk_len", type=int, default=768)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--spans_per_doc", type=int, default=3)
    ap.add_argument("--sents_per_claim", type=int, default=1)
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--use_deeplift", type=str, default="false")
    ap.add_argument("--refine_spans", type=str, default="true")
    # Retrieval via FlexiConc/FAISS (+SPLADE) → CE
    ap.add_argument("--flexiconc_db", type=str, default="/content/flexiconc.db",
                    help="Path to FlexiConc SQLite database with FAISS indices")
    ap.add_argument("--vector_backends", type=str,
                    default="all-mpnet-base-v2,BAAI/bge-base-en-v1.5",
                    help="Comma-separated dense encoders present in FlexiConc")
    ap.add_argument("--use_splade", type=str, default="false",
                    help="Enable SPLADE hybrid retrieval if available in adapter")
    ap.add_argument("--dense_weight", type=float, default=0.5,
                    help="Weight for dense score in dense+SPLADE fusion (pre-CE)")
    ap.add_argument("--splade_weight", type=float, default=0.5,
                    help="Weight for SPLADE score in dense+SPLADE fusion (pre-CE)")
    ap.add_argument("--ce_model", type=str,
                    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="CrossEncoder reranker name (Stage-2)")
    ap.add_argument("--ce_keep", type=int, default=100,
                    help="How many candidates to keep after CE (before IG)")
    ap.add_argument("--per_doc_cap", type=int, default=5,
                    help="Limit candidates per document pre-IG")
    args = ap.parse_args()

    use_deeplift = args.use_deeplift.lower().startswith("t")

    # --- Load row + inputs ---
    df = pd.read_csv(args.df_csv)
    if not (0 <= args.row_iloc < len(df)):
        raise SystemExit(f"Row {args.row_iloc} out of bounds for df length {len(df)}")
    row = df.iloc[args.row_iloc]

    # DOIs parsing (robust to str/list/NaN)
    dois_val = row.get(args.dois_col, None)
    import pandas as _pd
    if _pd.isna(dois_val):
        dois = []
    elif isinstance(dois_val, str):
        s = dois_val.strip()
        if s.startswith("["):
            try:
                dois = [str(x) for x in eval(s)]
            except Exception:
                dois = [t.strip() for t in s.split(",") if t.strip()]
        else:
            dois = [t.strip() for t in s.split(",") if t.strip()]
    elif isinstance(dois_val, (list, tuple)):
        dois = [str(x) for x in dois_val]
    else:
        dois = []
    dois = [d for d in dois if d]

    summary_text = str(row.get(args.summary_col, "")).strip()
    if not summary_text:
        raise SystemExit(f"No summary text in '{args.summary_col}'")

    # --- Load & index documents from src_dir for those DOIs ---
    src_dir = Path(args.src_dir)
    documents: Dict[str, str] = {}
    sentence_maps: Dict[str, Dict[str, Any]] = {}

    for doi in dois:
        fps = find_files_for_doi(doi, src_dir)
        if not fps:
            print(f"⚠️  No file found for DOI: {doi}")
            continue
        for fp in fps:
            try:
                raw = extract_text_from_pdf_robust(fp)
                cleaned = preprocess_pdf_text(raw)
                doc_id = f"{doi} :: {fp.name}"
                documents[doc_id] = cleaned
                sid2span, tree = build_sentence_index(cleaned)
                sentence_maps[doc_id] = {"sid2span": sid2span, "tree": tree}
                print(f"✅ Loaded {doc_id}: {len(cleaned)} chars, {len(sid2span)} sents")
            except Exception as e:
                print(f"❌ Failed {fp}: {e}")

    if not documents:
        raise SystemExit("No documents found/parsed.")

    # --- NLTK punkt (best-effort) ---
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try:
            nltk.download("punkt")
        except Exception:
            pass

    # --- Claims ---
    claims = split_summary_into_claims(summary_text) or [summary_text]

    # --- Retrieval + CE pruning (build compact doc texts from top sentences) ---
    pruned_documents = None
    try:
        pool = retrieve_candidates_for_claim(
            claim=summary_text,  # you can loop per-claim later if preferred
            flexiconc_db=args.flexiconc_db,
            vector_backends=[s.strip() for s in args.vector_backends.split(",") if s.strip()],
            use_splade=args.use_splade.lower().startswith("t"),
            dense_weight=args.dense_weight,
            splade_weight=args.splade_weight,
            ce_model=args.ce_model,
            ce_keep=args.ce_keep,
            per_doc_cap=args.per_doc_cap,
        )
        if pool:
            focused_docs: Dict[str, list] = {}
            valid_doc_ids = set(documents.keys())
            for it in pool:
                d = it.get("source_doc", "")
                s = it.get("sentence", "")
                if not s:
                    continue
                if d in valid_doc_ids:
                    focused_docs.setdefault(d, []).append(s)
                else:
                    # soft match (filename-only → match suffix)
                    matches = [doc for doc in valid_doc_ids if doc.endswith(d) or d in doc]
                    if matches:
                        focused_docs.setdefault(matches[0], []).append(s)
            if focused_docs:
                pruned_documents = {d: "\n".join(sents) for d, sents in focused_docs.items()}
                print(f"🔎 Pruned to {len(pruned_documents)} focused docs using retrieval+CE")
        else:
            print("ℹ️ Retrieval returned no pool; proceeding with full documents.")
    except Exception as e:
        print(f"ℹ️ Retrieval prune unavailable: {e}. Proceeding with full documents.")

    # ---- ALWAYS continue: choose active_documents and run RA ----
    active_documents = pruned_documents if pruned_documents else documents
    doc_ids = list(active_documents.keys())
    print(f"📚 Active documents for RA: {len(doc_ids)}")

    # --- RA engine ---
    ra = Seq2SeqRevAttrIG(model_name=args.model_name, ig_steps=args.ig_steps,
                          use_deeplift=use_deeplift)

    # --- Run RA per-claim ---
    per_claim_results, flat_rows = [], []
    for cid, claim in enumerate(claims, start=1):
        print(f"\n🧭 Claim {cid}/{len(claims)}: {claim[:120]}")
        spans = ra.find_evidence_spans_for_summary(
            claim,
            [active_documents[d] for d in doc_ids],
            chunk_len=args.chunk_len,
            stride=args.stride,
            spans_per_doc=args.spans_per_doc
        )
        sent_hits = map_spans_to_sentences(
            spans, active_documents, sentence_maps, doc_ids,
            top_sentences_per_claim=args.sents_per_claim
        )
        per_claim_results.append({
            "claim_id": cid,
            "claim_text": claim,
            "evidence_sentences": [{
                "source_doc": h["source_doc"],
                "sentence_idx": h["sentence_idx"],
                "sentence": h["sentence"],
                "doc_char_start": h["doc_char_start"],
                "doc_char_end": h["doc_char_end"],
                "span_char_start": h["span_char_start"],
                "span_char_end": h["span_char_end"],
                "span_score": h["span_score"],
            } for h in sent_hits]
        })
        for h in sent_hits:
            flat_rows.append({
                "claim_id": cid,
                "claim_text": claim,
                "source_doc": h["source_doc"],
                "sentence_idx": h["sentence_idx"],
                "sentence": h["sentence"],
                "span_score": h["span_score"],
                "doc_char_start": h["doc_char_start"],
                "doc_char_end": h["doc_char_end"],
                "span_char_start": h["span_char_start"],
                "span_char_end": h["span_char_end"],
            })

    # --- Outputs ---
    out_json = f"{args.out_prefix}_seq2seq_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary_text, "total_claims": len(claims), "results": per_claim_results},
            f, ensure_ascii=False, indent=2
        )
    print(f"💾 Saved: {out_json}")

    out_csv = f"{args.out_prefix}_seq2seq_evidence.csv"
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"💾 Saved: {out_csv}")


if __name__ == "__main__":
    main()