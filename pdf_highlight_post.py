# pdf_highlight_post.py
"""
Postprocessing: turn ENLENS/SpanBERT production_output.json
into an Adobe-compatible highlighted PDF.

We support 2 highlight types:
- classifier spans (SDG / value / whatever the sentence-span classifier emitted)
- coreference mentions (per chain)

Collision policy:
1. dedupe per type (IOU-based merge)
2. draw classifier first (yellow, 0.35)
3. draw coref second (chain color, 0.25)
4. optionally skip coref if fully covered by classifier

Usage:
    python pdf_highlight_post.py analysis.json output.pdf
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Tuple
import fitz  # PyMuPDF


# ------------------------------------------------------------
# geometry utils
# ------------------------------------------------------------
def iou(b1: Dict[str, float], b2: Dict[str, float]) -> float:
    x_left = max(b1["l"], b2["l"])
    y_top = max(b1["t"], b2["t"])
    x_right = min(b1["r"], b2["r"])
    y_bottom = min(b1["b"], b2["b"])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1["r"] - b1["l"]) * (b1["b"] - b1["t"])
    area2 = (b2["r"] - b2["l"]) * (b2["b"] - b2["t"])
    return inter / float(area1 + area2 - inter)


def merge_same_type(highlights: List[Dict[str, Any]], iou_thr: float = 0.6) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for h in highlights:
        placed = False
        for m in merged:
            if h["page"] == m["page"] and iou(h["bbox"], m["bbox"]) >= iou_thr:
                # union
                m["bbox"]["l"] = min(m["bbox"]["l"], h["bbox"]["l"])
                m["bbox"]["t"] = min(m["bbox"]["t"], h["bbox"]["t"])
                m["bbox"]["r"] = max(m["bbox"]["r"], h["bbox"]["r"])
                m["bbox"]["b"] = max(m["bbox"]["b"], h["bbox"]["b"])
                placed = True
                break
        if not placed:
            merged.append(h)
    return merged


def docling_bbox_to_fitz_rect(bbox: Dict[str, float], page_height: float) -> fitz.Rect:
    """
    Docling/pypdf-style bbox: (l, t, r, b) with TOP-LEFT origin
    PyMuPDF: bottom-left origin
    """
    l, t, r, b = bbox["l"], bbox["t"], bbox["r"], bbox["b"]
    x0 = l
    y0 = page_height - b
    x1 = r
    y1 = page_height - t
    return fitz.Rect(x0, y0, x1, y1)


def get_chain_color(chain_id: int) -> Tuple[float, float, float]:
    import colorsys
    # golden-ratio spacing for nice distinct colors
    hue = (chain_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    return (r, g, b)


def is_covered(coref_hl: Dict[str, Any], clf_hls: List[Dict[str, Any]], iou_thr: float = 0.85) -> bool:
    for c in clf_hls:
        if c["page"] == coref_hl["page"] and iou(c["bbox"], coref_hl["bbox"]) >= iou_thr:
            return True
    return False


# ------------------------------------------------------------
# main entry
# ------------------------------------------------------------
def render_highlights_from_json(
    analysis_json: str,
    output_pdf: str,
    min_cls_conf: float = 0.7,
    skip_fully_covered_coref: bool = True,
) -> int:
    with open(analysis_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    src_pdf = data["source_pdf"]
    doc = fitz.open(src_pdf)

    # --------------------------------------------------------
    # 1. collect classifier highlights
    # --------------------------------------------------------
    clf_hls: List[Dict[str, Any]] = []
    for sent in data.get("sentence_analyses", []):
        cls = sent.get("classification") or {}
        conf = cls.get("confidence") or cls.get("score") or 0.0
        if conf < min_cls_conf:
            continue

        # try sentence-level bbox first
        page_no = sent.get("page_no")
        bbox = sent.get("bbox")

        # fallback to first span in that sentence
        if not bbox and sent.get("span_analysis"):
            first_span = sent["span_analysis"][0]
            page_no = first_span.get("page_no", page_no)
            bbox = first_span.get("bbox", bbox)

        if not bbox or page_no is None:
            continue

        clf_hls.append({
            "type": "classifier",
            "page": int(page_no),
            "bbox": {
                "l": float(bbox["l"]),
                "t": float(bbox["t"]),
                "r": float(bbox["r"]),
                "b": float(bbox["b"]),
            },
            "label": f"SDG {cls.get('label', '?')}",
            "text": sent.get("sentence_text", "")[:160],
            "confidence": float(conf),
        })

    clf_hls = merge_same_type(clf_hls, iou_thr=0.6)

    # --------------------------------------------------------
    # 2. collect coref highlights
    # --------------------------------------------------------
    coref_hls: List[Dict[str, Any]] = []
    coref = data.get("coreference_analysis") or {}
    for chain in coref.get("chains", []):
        cid = int(chain.get("chain_id", -1))
        for m in chain.get("mentions", []):
            bbox = m.get("bbox")
            page_no = m.get("page_no")
            if not bbox or page_no is None:
                continue
            coref_hls.append({
                "type": "coref",
                "page": int(page_no),
                "bbox": {
                    "l": float(bbox["l"]),
                    "t": float(bbox["t"]),
                    "r": float(bbox["r"]),
                    "b": float(bbox["b"]),
                },
                "chain_id": cid,
                "text": m.get("text", "")[:120],
                "rep": chain.get("representative", ""),
            })

    coref_hls = merge_same_type(coref_hls, iou_thr=0.6)

    # --------------------------------------------------------
    # 3. write
    # --------------------------------------------------------
    wrote = 0

    # 3a) classifier first
    for hl in clf_hls:
        page = doc[hl["page"]]
        rect = docling_bbox_to_fitz_rect(hl["bbox"], page.rect.height)
        annot = page.add_highlight_annot(rect)
        annot.set_colors(stroke=(1.0, 1.0, 0.0))  # yellow
        annot.set_opacity(0.35)
        annot.set_info({
            "title": "ENLENS",
            "subject": hl["label"],
            "content": f"conf={hl['confidence']:.2f}\n{hl['text']}",
        })
        annot.update()
        wrote += 1

    # 3b) coref next
    for hl in coref_hls:
        if skip_fully_covered_coref and is_covered(hl, clf_hls, iou_thr=0.85):
            continue
        page = doc[hl["page"]]
        rect = docling_bbox_to_fitz_rect(hl["bbox"], page.rect.height)
        color = get_chain_color(hl["chain_id"])
        annot = page.add_highlight_annot(rect)
        annot.set_colors(stroke=color)
        annot.set_opacity(0.25)
        annot.set_info({
            "title": "ENLENS coref",
            "subject": f"Chain {hl['chain_id']}",
            "content": f"{hl['rep']}\n{hl['text']}",
        })
        annot.update()
        wrote += 1

    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()
    return wrote


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_json", help="JSON file produced by ENLENS_SpanBert_corefree_prod")
    ap.add_argument("output_pdf", help="Where to save highlighted PDF")
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--no-skip-covered", action="store_true")
    args = ap.parse_args()

    wrote = render_highlights_from_json(
        args.analysis_json,
        args.output_pdf,
        min_cls_conf=args.min_conf,
        skip_fully_covered_coref=not args.no_skip_covered,
    )
    print(f"✓ wrote {wrote} annotations to {args.output_pdf}")
