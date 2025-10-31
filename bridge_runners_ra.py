# -*- coding: utf-8 -*-
# bridge_runners_ra.py

from typing import Dict, Any, List, Tuple, Optional, Callable
import importlib
import logging
import inspect
from pathlib import Path

from helper import encode_mpnet, encode_sdg_hidden, encode_scico
from helper_addons import NGramIndex
try:
    from helper_addons import build_ngram_trie, build_cooc_graph
except ImportError:
    build_ngram_trie = None
    build_cooc_graph = None

from ENLENS_SpanBert_corefree_prod import run_quick_analysis as _run_quick_span
try:
    from ENLENS_KP_BERT_corefree_prod import run_quick_analysis as _run_quick_kpe
except Exception:
    _run_quick_kpe = None

from flexiconc_adapter import (
    export_production_to_flexiconc,
    open_db,
    upsert_doc_cooc,
    upsert_doc_trie,
)

# NEW: reuse original TXT writer to stage a text corpus (optional)
from make_flexiconc_db import write_text_from_pdf

# NEW: build FAISS indices after ingestion (optional)
from flexiconc_adapter import build_faiss_indices

DEFAULT_EMBEDDING_MODELS = {
    "mpnet": encode_mpnet,
    "sdg-bert": encode_sdg_hidden,
    "scico": encode_scico,
}

logger = logging.getLogger(__name__)


def _unify_sentence_fields(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce normalized lists the UI can always rely on:
      - span_terms: list[str]
      - kp_terms:   list[str]
      - token_terms:list[str]
    Leaves original fields intact.
    """
    out = s

    # ---- spans (several possible names across your variants)
    span_blocks = (
        s.get("span_analysis")
        or s.get("spans")
        or s.get("mask_spans")
        or []
    )
    span_terms: List[str] = []
    for it in span_blocks:
        if isinstance(it, dict):
            t = it.get("text") or it.get("span") or it.get("token") or ""
        else:
            t = str(it)
        t = t.strip()
        if t:
            span_terms.append(t)
    out["span_terms"] = span_terms

    # ---- keyphrases (several shapes)
    kp = s.get("keyphrases") or s.get("kpe") or s.get("kp") or []
    if isinstance(kp, dict):
        # common shapes: {"phrases":[{"text":..},..]} or {"topk":[...]}
        kp = kp.get("phrases") or kp.get("topk") or []
    kp_terms: List[str] = []
    for it in kp:
        if isinstance(it, dict):
            t = it.get("text") or it.get("phrase") or ""
        else:
            t = str(it)
        t = t.strip()
        if t:
            kp_terms.append(t)
    out["kp_terms"] = kp_terms

    # ---- tokens (for fallback chips)
    toks = (s.get("token_analysis") or {}).get("tokens") or []
    token_terms: List[str] = []
    for it in toks:
        if isinstance(it, dict):
            t = it.get("token") or it.get("text") or ""
        else:
            t = str(it)
        t = t.strip()
        if t:
            token_terms.append(t)
    out["token_terms"] = token_terms

    return out


def _only_supported_kwargs(fn, **maybe_kwargs):
    """Keep kwargs present in fn signature; if fn has **kwargs, allow all."""
    sig = inspect.signature(fn)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return {k: v for k, v in maybe_kwargs.items()}
    allowed = {p.name for p in sig.parameters.values()}
    return {k: v for k, v in maybe_kwargs.items() if k in allowed}


def _unwrap_production(obj):
    """Accept bare production dict or {'production_output': {...}}."""
    if not isinstance(obj, dict):
        return None
    if "sentence_analyses" in obj and "full_text" in obj:
        return obj
    if "production_output" in obj and isinstance(obj["production_output"], dict):
        return obj["production_output"]
    return None

from inspect import signature

def call_build_cooc_graph(build_cooc_graph, sentences, **kw):
    """
    Calls build_cooc_graph regardless of whether it expects
    a 'sentences=' kwarg (new API) or a single text (old API).
    """
    sig = signature(build_cooc_graph)
    if "sentences" in sig.parameters:
        mapped = dict(
            sentences=sentences,
            window_size=kw.get("window_size", kw.get("window", 5)),
            min_count=kw.get("min_count", 2),
            topk_neighbors=kw.get("topk_neighbors", 20),
            mode=kw.get("mode", "spacy"),           # or "hf"
            hf_tokenizer=kw.get("hf_tokenizer"),
            cooc_impl=kw.get("cooc_impl", "fast"),
            cooc_max_tokens=kw.get("cooc_max_tokens", 50000),
            cache_key=kw.get("cache_key"),
        )
        return build_cooc_graph(**{k: v for k, v in mapped.items() if k in sig.parameters})
    else:
        # old API: first arg is a string with the whole text
        text = "\n".join(sentences)
        mapped = dict(
            window=kw.get("window_size", kw.get("window", 5)),
            min_count=kw.get("min_count", 2),
            topk_neighbors=kw.get("topk_neighbors", 20),
            mode=kw.get("mode", "spacy"),
            hf_tokenizer=kw.get("hf_tokenizer"),
            cooc_impl=kw.get("cooc_impl", "fast"),
            cooc_max_tokens=kw.get("cooc_max_tokens", 50000),
            cache_key=kw.get("cache_key"),
        )
        return build_cooc_graph(text, **{k: v for k, v in mapped.items() if k in sig.parameters})

def run_ingestion_quick(
    pdf_path: str,
    *,
    # top-level size limits
    max_sentences: int | None = None,
    max_text_length: int | None = None,
    # pipeline selection
    candidate_source: str | None = None,  # "span" or "kp"
    # explainability / spans / KPE
    ig_enabled: bool | None = None,
    span_masking_enabled: bool | None = None,
    max_span_len: int | None = None,
    top_k_spans: int | None = None,
    kpe_top_k: int | None = None,
    kpe_threshold: float | None = None,
    # fastcoref knobs
    coref_scope: str | None = None,            # 'whole_document' | 'windowed'
    coref_window_sentences: int | None = None,
    coref_window_stride: int | None = None,
    coref_backend: str | None = None,
    coref_device: str | None = None,
    resolve_text: bool | None = None,
    # sdg / consensus (if exposed)
    agree_threshold: float | None = None,
    disagree_threshold: float | None = None,
    min_confidence: float | None = None,
    # co-occurrence knobs
    cooc_mode: str | None = None,              # "spacy" | "hf" | "none"
    cooc_hf_tokenizer: Any | None = None,
    # anything else the pipeline might take in the future
    **extra,
):
    """
    UI → bridge → pipeline with safe kwarg filtering and pipeline selection.

    Returns:
        {"ok": True/False, "production_output": dict, "_warn": [..]} (on success),
        or {"ok": False, "error": "..."} (on failure).
    """
    pdf_path = str(pdf_path)
    if not Path(pdf_path).exists():
        return {"ok": False, "error": f"file not found: {pdf_path}"}

    # Back-compat for callers passing these via **extra
    legacy_mode = extra.pop("cooc_mode", None)
    legacy_tok = extra.pop("cooc_hf_tokenizer", None)
    if cooc_mode is None:
        cooc_mode = legacy_mode
    if cooc_hf_tokenizer is None:
        cooc_hf_tokenizer = legacy_tok

    # --- Select pipeline
    if candidate_source == "kp" and _run_quick_kpe is not None:
        pipeline_fn = _run_quick_kpe
        if "shortlist_mode" in extra and "coref_shortlist_mode" not in extra:
            extra["coref_shortlist_mode"] = extra.pop("shortlist_mode")
        
        raw_kwargs = dict(
            pdf_file=pdf_path,  # KPE uses pdf_file
            max_sentences=max_sentences,
            max_text_length=max_text_length,
            top_k_phrases=kpe_top_k or top_k_spans,  # KPE name
            kpe_threshold=kpe_threshold,
            agree_threshold=agree_threshold,
            disagree_threshold=disagree_threshold,
            min_confidence=min_confidence,
            **extra,
        )
    else:
        # Default to SpanBERT pipeline
        pipeline_fn = _run_quick_span
        raw_kwargs = dict(
            pdf_file=pdf_path,  # SpanBERT also expects pdf_file in your codebase
            max_sentences=max_sentences,
            max_text_length=max_text_length,
            coref_scope=coref_scope,
            coref_window_sentences=coref_window_sentences,
            coref_window_stride=coref_window_stride,
            coref_backend=coref_backend,
            coref_device=coref_device,
            resolve_text=resolve_text,
            agree_threshold=agree_threshold,
            disagree_threshold=disagree_threshold,
            min_confidence=min_confidence,
        )
        # Only pass supported extras
        _whitelist = {
            "coref_shortlist_mode",
            "coref_shortlist_topk",
            "coref_trie_tau",
            "coref_cooc_tau",
            "coref_use_pair_scorer",
            "coref_scorer_threshold",
            "coref_pair_scorer",
            # co-occ builder knobs
            "cooc_window",
            "cooc_min_count",
            "cooc_topk_neighbors",
            "cooc_mode",
            "cooc_hf_tokenizer",
        }
        for k in _whitelist:
            if k in extra and extra[k] is not None:
                raw_kwargs[k] = extra[k]

    # Drop None values to avoid overriding pipeline defaults
    raw_kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}

    # Keep only arguments that the pipeline really accepts
    pipe_kwargs = _only_supported_kwargs(pipeline_fn, **raw_kwargs)
    if "coref_shortlist_mode" in pipe_kwargs:
        print(f"[bridge] coref_shortlist_mode -> {pipe_kwargs['coref_shortlist_mode']}")

    # --- Call selected pipeline
    result = pipeline_fn(**pipe_kwargs)

    # --- Normalize: always put production dict in "production_output"
    prod = _unwrap_production(result) or result
    if not isinstance(prod, dict):
        return {"ok": False, "error": "Pipeline returned unexpected object."}

    # Attach a lightweight coref n-gram index (non-fatal on failure)
    try:
        from helper_addons import build_ngram_index
        chains = (prod.get("coreference_analysis") or {}).get("chains") or []
        if chains:
            idx_obj = build_ngram_index(chains, char_n=4, token_ns=(2, 3), build_trie=False)
            payload = {
                "chain_grams": idx_obj.chain_grams,  # dict[int -> Counter]
                "idf": idx_obj.idf,                  # dict[str -> float]
            }
            prod.setdefault("indices", {})["coref_ngram"] = payload
    except Exception as e:
        if isinstance(result, dict):
            result.setdefault("_warn", []).append(f"could_not_attach_coref_ngram_index: {e}")

    # --- Optional: export to FlexiConc
    db_path = extra.get("flexiconc_db_path")
    doc_id = extra.get("doc_id") or Path(pdf_path).stem
    if db_path:
        # 1) Export production (sentences + embeddings)
        export_production_to_flexiconc(
            db_path,
            doc_id,
            prod,
            uri=pdf_path,
            embedding_models=DEFAULT_EMBEDDING_MODELS,
        )

        # 2) Export trie + co-occurrence indices (best-effort)
        _build_ngram_trie = globals().get("build_ngram_trie")
        _build_cooc_graph = globals().get("build_cooc_graph")
        if callable(_build_ngram_trie) and callable(_build_cooc_graph):
        
            cx = open_db(db_path)
            try:
                # Build trie from coref chains
                chains = (prod.get("coreference_analysis") or {}).get("chains") or []
                trie_root, trie_idf, chain_grams = _build_ngram_trie(
                    chains,
                    char_n=4,
                    token_ns=(2, 3),
                )
                upsert_doc_trie(cx, doc_id, trie_root, trie_idf, chain_grams)
                print("  ✅ trie saved")

                # Pull co-occ sentences from production
                sentences_for_cooc = (
                    prod.get("filtered_sentences")
                    or prod.get("sentences")
                    or prod.get("doc_sentences")
                    or [s.get("sentence_text","") for s in (prod.get("sentence_analyses") or [])]
                    or []
                )
                sentences_for_cooc = [s for s in sentences_for_cooc if isinstance(s, str) and s.strip()]
                if not sentences_for_cooc:
                    print("  ℹ️ no sentences available for co-occurrence; skipping")
                else:
                    # Configure co-occ builder
                    effective_mode = (cooc_mode or "spacy").lower()
                    if effective_mode == "none":
                        print("  ℹ️ co-occ disabled by flag")
                    else:
                        effective_tokenizer = cooc_hf_tokenizer if effective_mode == "hf" else None
                        if effective_mode == "hf" and effective_tokenizer is None:
                            msg = ("cooc_mode='hf' requested but no tokenizer supplied; "
                                   "falling back to spaCy for co-occurrence export.")
                            print(f"  ⚠️ {msg}")
                            effective_mode = "spacy"
                            if isinstance(result, dict):
                                result.setdefault("_warn", []).append(msg)
                        
                        rows = None
                        row_norms = None
                        vocab = {}
                        try:
                             # Ensure kwarg names match your build_cooc_graph signature
                             cooc_window = extra.get("cooc_window", 5)
                             cooc_min_count = extra.get("cooc_min_count", 2)
                             cooc_topk = extra.get("cooc_topk_neighbors", 10)
                             effective_impl = extra.get("cooc_impl", "fast")
                             cooc_max_tokens = int(extra.get("cooc_max_tokens", 50000))
                             cache_key = f"{doc_id}:{len(sentences_for_cooc)}"
                             effective_impl = extra.get("cooc_impl", "fast")
                             vocab, rows, row_norms = call_build_cooc_graph(                                 
                                 _build_cooc_graph,
                                 sentences=sentences_for_cooc,
                                 window_size=cooc_window,
                                 min_count=cooc_min_count,
                                 topk_neighbors=cooc_topk,
                                 mode=effective_mode,
                                 hf_tokenizer=cooc_hf_tokenizer,
                                 cooc_impl=effective_impl,
                                 cooc_max_tokens=cooc_max_tokens,
                                 cache_key=cache_key,
                              )    
                        
                        
                        except Exception as e:
                            print(f"  ⚠️ Skipping co-occ build: {e}")
                            rows = None
                    
                        if rows is not None:
                             # Coerce to CSR; else skip
                             try:
                                from scipy.sparse import csr_matrix, issparse
                                if not issparse(rows) or not hasattr(rows, "getformat") or rows.getformat() != "csr":
                                    rows = csr_matrix(rows)
                             except Exception as e:
                                 print(f"  ⚠️ Skipping co-occ export (no CSR): {e}")
                                 rows = None

                        if rows is not None:
                            try:
                                upsert_doc_cooc(cx, doc_id, vocab, rows, row_norms)
                                print("  ✅ co-occurrence saved")
                            except Exception as e:
                                print(f"  ⚠️ co-occ upsert failed; continuing: {e}")
                        else:
                            print("  ℹ️ co-occ rows unavailable; continuing without co-occurrence.")
            finally:
                cx.close()
        else:
            print("  ℹ️ Skipping trie/co-occ export: helper_addons missing required builders.")

    return {"ok": True, "production_output": prod}


def rows_from_production(production_output: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not production_output:
        return []
    rows = []
    for s in (production_output.get("sentence_analyses") or []):
        rows.append(dict(
            sid=s.get("sentence_id"),
            text=s.get("sentence_text"),
            path="",
            start=s.get("doc_start"),
            end=s.get("doc_end"),
        ))
    return rows


def build_scico(
    rows: List[Dict[str, Any]],
    selected_terms: List[str],
    scico_cfg: Dict[str, Any],
    *,
    precomputed_embeddings: Any = None,
    embedding_provider: Optional[Callable[[List[str]], Any]] = None,
):
    """
    Wraps scico_graph_pipeline.build_graph_from_selection with the UI dict.
    """
    mod = importlib.import_module("scico_graph_pipeline")
    build = getattr(mod, "build_graph_from_selection")
    ScicoConfig = getattr(mod, "ScicoConfig")

    coherence_opts = dict(
        faiss_topk=scico_cfg["faiss_topk"],
        nprobe=scico_cfg["nprobe"],
        add_lsh=scico_cfg["add_lsh"],
        lsh_threshold=scico_cfg["lsh_threshold"],
        minhash_k=scico_cfg["minhash_k"],
        cheap_len_ratio=scico_cfg["cheap_len_ratio"],
        cheap_jaccard=scico_cfg["cheap_jaccard"],
        use_coherence=scico_cfg["use_coherence"],
        coherence_threshold=scico_cfg["coherence_threshold"],
        max_pairs=scico_cfg["max_pairs"],
    )

    G, meta = build(
        rows,
        selected_terms=selected_terms,
        kmeans_k=scico_cfg["kmeans_k"],
        clustering_method=scico_cfg["clustering"],
        community_on=scico_cfg["community_on"],
        community_method=scico_cfg["community_method"],
        community_weight="prob",
        scico_cfg=ScicoConfig(prob_threshold=scico_cfg["prob_threshold"]),
        add_layout=True,
        use_coherence_shortlist=scico_cfg["use_shortlist"],
        coherence_opts=coherence_opts,
        max_degree=scico_cfg["max_degree"],
        top_edges_per_node=scico_cfg["top_edges_per_node"],
        summarize=scico_cfg["summarize"],
        summarize_on=scico_cfg["summarize_on"],
        summary_methods=scico_cfg["summary_methods"],
        summary_opts=dict(
            num_sentences=scico_cfg["xsum_sentences"],
            sdg_targets=None,
            sdg_top_k=scico_cfg["sdg_topk"],
            cross_encoder_model=scico_cfg["cross_encoder_model"] or None,
            centroid_sim_threshold=scico_cfg["centroid_sim_threshold"],
            centroid_top_n=scico_cfg["centroid_top_n"],
            centroid_store_vector=scico_cfg["centroid_store_vector"],
        ),
        precomputed_embeddings=precomputed_embeddings,
        embedding_provider=embedding_provider,
    )
    return G, meta


def build_concordance(
    sqlite_path: str,
    terms: List[str],
    and_mode: bool = True,
    *,
    vector_backend: Optional[str] = None,
    use_faiss: bool = True,
    limit: int = 500,
) -> Dict[str, Any]:
    """Run concordance query with optional vector backend + FAISS fallback."""
    normalized_terms = [t for t in terms if t and str(t).strip()]
    fallback = {
        "rows": [],
        "embeddings": None,
        "meta": {
            "mode": "unavailable",
            "vector_backend": vector_backend,
            "faiss_used": False,
            "query_text": "",
            "total_candidates": 0,
            "scores": [],
        },
        "terms": normalized_terms,
    }
    try:
        mod = importlib.import_module("flexiconc_adapter")
        search = getattr(mod, "query_concordance", None)
        if search is None:
            return fallback
        result = search(
            sqlite_path,
            normalized_terms,
            mode="AND" if and_mode else "OR",
            limit=int(limit),
            vector_backend=vector_backend,
            use_faiss=use_faiss,
        )
        if isinstance(result, dict):
            return result
        return {
            "rows": list(result or []),
            "embeddings": None,
            "meta": {
                "mode": "legacy",
                "vector_backend": vector_backend,
                "faiss_used": False,
                "query_text": " ".join(normalized_terms),
                "total_candidates": len(result or []),
                "scores": [],
            },
            "terms": normalized_terms,
        }
    except Exception as exc:
        logger.warning("concordance query failed: %s", exc)
        return fallback


def pick_sentence_coref_groups(production_output: Dict[str, Any], sent_idx: int):
    sents = production_output.get("sentence_analyses") or []
    try:
        idx_payload = (production_output.get("indices") or {}).get("coref_ngram")
    except Exception:
        idx_payload = None

    # Pre-compute sentence spans so we can fall back to doc offsets if needed.
    sent_spans: List[Tuple[Optional[int], Optional[int]]] = []
    for sent in sents:
        start = sent.get("doc_start")
        end = sent.get("doc_end")
        try:
            start = int(start) if start is not None else None
        except (TypeError, ValueError):
            start = None
        try:
            end = int(end) if end is not None else None
        except (TypeError, ValueError):
            end = None
        sent_spans.append((start, end))

    if idx_payload:
        ng_index = NGramIndex.from_dict(idx_payload)
    else:
        chains = (production_output.get("coreference_analysis") or {}).get("chains") or []
        if not chains:
            ng_index = NGramIndex.from_dict({"chain_grams": {}, "idf": {}})
        else:
            from helper_addons import build_ngram_index
            ng_index = build_ngram_index(
                chains,
                char_n=4,
                token_ns=(2, 3),
                build_trie=True
            )

    ng_index.ensure_trie()

    if not (production_output.get("coreference_analysis") or {}).get("chains"):
        return {}
    chains = production_output["coreference_analysis"]["chains"]

    if sent_idx is None or sent_idx >= len(sents):
        return {}

    out = {}
    for chain in chains:
        mentions = chain.get("mentions") or []
        edges    = chain.get("edges") or []
        # map anaphor index -> tag (pick first if multiple)
        tag_by_anaph = {int(e["anaphor"]): e.get("tag")
                        for e in edges if isinstance(e, dict) and "anaphor" in e}

        # group chain mentions by sentence id for display
        sent_map: dict[int, dict] = {}
        for i, m in enumerate(mentions):
            sid = m.get("sent_id")
            if sid is None or sid >= len(sents):
                continue
            if sid is not None:
                try:
                    sid = int(sid)
                except (TypeError, ValueError):
                    sid = None
            if sid is None or sid < 0 or sid >= len(sents):
                # Attempt to locate the sentence by absolute char offsets.
                start_char = m.get("start_char")
                end_char = m.get("end_char")
                if start_char is None or end_char is None:
                    start_char = m.get("start")
                    end_char = m.get("end")
                try:
                    start_char = int(start_char) if start_char is not None else None
                except (TypeError, ValueError):
                    start_char = None
                try:
                    end_char = int(end_char) if end_char is not None else None
                except (TypeError, ValueError):
                    end_char = None

                if start_char is not None and end_char is not None:
                    for idx, (span_start, span_end) in enumerate(sent_spans):
                        if span_start is None or span_end is None:
                            continue
                        if span_start <= start_char and end_char <= span_end:
                            sid = idx
                            m["sent_id"] = sid
                            break

                if sid is None or sid < 0 or sid >= len(sents):
                    continue
            rec = sent_map.setdefault(sid, {
                "sid": sid,
                "text": sents[sid].get("sentence_text", ""),
                "mentions": []
            })
            rec["mentions"].append({
                "text": m.get("text", ""),
                "start_char": m.get("start_char"),
                "end_char": m.get("end_char"),
                "tag": tag_by_anaph.get(i)  # may be None for first/standalone
            })

        # only include chains that touch the selected sentence
        if sent_idx in sent_map:
            out[int(chain.get("chain_id", 0))] = list(sent_map.values())
    return out


def compute_sentence_clusters_for_doc(G, meta, sent_idx: int):
    # Placeholder if you later want to compute detailed cluster membership lists
    return {}


# ---------------------------
# Runner / CLI
# ---------------------------

def _gather_pdfs(src_dir: str | None, pdf: str | None, recursive: bool) -> List[Path]:
    pdfs: List[Path] = []
    if pdf:
        p = Path(pdf)
        if not p.exists():
            raise SystemExit(f"file not found: {p}")
        pdfs.append(p)
    if src_dir:
        root = Path(src_dir)
        if not root.exists():
            raise SystemExit(f"directory not found: {root}")
        pattern = "**/*.pdf" if recursive else "*.pdf"
        found = sorted(root.glob(pattern))
        if not found:
            print(f"⚠️  No PDFs found under {root} (pattern={pattern})")
        pdfs.extend(found)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in pdfs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Ingest PDFs into FlexiConc (DB) and optionally export plain-text corpus."
    )
    # input selection
    ap.add_argument("--pdf", help="Path to a single PDF to ingest")
    ap.add_argument("--src_dir", help="Directory of PDFs to ingest")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders for --src_dir")

    # core actions
    ap.add_argument("--build", action="store_true", help="Run ingestion/export")
    ap.add_argument("--db", type=str, default="flexiconc.db", help="Output FlexiConc SQLite DB")
    ap.add_argument("--faiss", action="store_true", help="(Re)build FAISS indices after ingestion")

    # TXT export (original FlexiConc behavior)
    ap.add_argument("--export_txt_root", type=str,
                    help="If set, also write .txt files into this folder (e.g., /content/export)")
    ap.add_argument("--lang", type=str, default="en", choices=["en", "pl"],
                    help="Language bucket for TXT export (subfolder under export root)")

    # passthrough knobs to run_ingestion_quick(...)
    ap.add_argument("--candidate_source", type=str, default=None, help="'span' or 'kp'")
    ap.add_argument("--max_sentences", type=int, default=None)
    ap.add_argument("--max_text_length", type=int, default=None)

    args = ap.parse_args()

    if not args.build:
        ap.print_help()
        return

    pdfs = _gather_pdfs(args.src_dir, args.pdf, args.recursive)
    if not pdfs:
        raise SystemExit("Nothing to ingest. Provide --pdf and/or --src_dir.")

    print(f"🏗️ Ingesting {len(pdfs)} PDF(s) → {args.db}")
    ok_count, fail_count = 0, 0

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] • {pdf_path}")

        # 1) (optional) write .txt to export tree, like original FlexiConc scripts
        if args.export_txt_root:
            try:
                out_txt = write_text_from_pdf(Path(args.export_txt_root), Path(pdf_path), lang=args.lang)
                print(f"  📝 wrote TXT: {out_txt}")
            except Exception as e:
                # TXT export is best-effort and should not block DB ingestion
                print(f"  ⚠️ TXT export failed; continuing: {e}")

        # 2) ingest into DB via your existing quick pipeline
        try:
            result = run_ingestion_quick(
                str(pdf_path),
                flexiconc_db_path=args.db,
                doc_id=Path(pdf_path).stem,
                candidate_source=args.candidate_source,
                max_sentences=args.max_sentences,
                max_text_length=args.max_text_length,
            )
            if not isinstance(result, dict) or not result.get("ok"):
                fail_count += 1
                print(f"  ❌ Ingestion failed: {result.get('error') if isinstance(result, dict) else 'unknown error'}")
            else:
                ok_count += 1
                print("  ✅ Ingested")
        except Exception as e:
            fail_count += 1
            print(f"  ❌ Exception: {e}")

    print(f"🏁 Done. Success: {ok_count} | Failed: {fail_count}")

    # 3) Build FAISS indices (if requested)
    if args.faiss:
        try:
            cx = open_db(args.db)
            cur = cx.cursor()
            rows = cur.execute("SELECT DISTINCT model FROM embeddings").fetchall()
            models = [r[0] for r in rows if r and r[0]]
            if not models:
                print("⚠️  No embeddings found; FAISS build skipped.")
            else:
                print(f"🗂️  Building FAISS indices for models: {models}")
                build_faiss_indices(cx, models=models)  # correct kw
                print("✅ FAISS indices built.")
        except Exception as e:
            print(f"⚠️  Could not build FAISS indices: {e}")
        finally:
            try:
                cx.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()


"""
Created on Sat Oct 25 20:29:33 2025
@author: niran
"""
