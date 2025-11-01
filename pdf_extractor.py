"""
pdf_extractor.py

Multistage PDF → text (and page-map) extractor for the ENLENS / cross-doc pipeline.

Order of fallbacks:
1. Docling proper
2. pypdf → Docling datamodel (synthetic Docling)  ← keeps per-page structure
3. pypdf + cleaning (string)                      ← lightweight
4. PyPDF2                                         ← last resort

All current callers can keep doing:
    text = extract_text_from_pdf_robust(path)

For richer callers (classifier/coref alignment), do:
    doc = extract_text_from_pdf_robust(path, return_doc=True, clean_cites=True, clean_artifacts=True)
    text = doc["text"]
    pages = doc["pages"]   # [{page_no, original, cleaned}, ...]

This file is self-contained except for optional docling and pypdf deps.
"""

from __future__ import annotations

import re
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------
# optional imports
# ---------------------------------------------------------
try:
    from pypdf import PdfReader as _PYPDF_READER
except Exception:
    _PYPDF_READER = None


# =========================================================
# 1. Cleaning helpers
# =========================================================
def _clean_citations(
    text: str,
    remove_brackets: bool = True,
    remove_superscript_numbers: bool = True,
) -> str:
    """
    Remove common citation patterns:
    - [1], [1,2], [1-5], [Author et al., 2024], [Author, 2024]
    - superscript digits
    - (1), (2,3) but not (2024)
    """
    cleaned = text

    if remove_brackets:
        # [1], [1,2], [1-3], [1,2,3-5]
        cleaned = re.sub(r'\[\d+(?:[,\-]\s*\d+)*\]', '', cleaned)
        # [Author et al., 2024]
        cleaned = re.sub(r'\[[\w\s]+et al\.,?\s*\d{4}\]', '', cleaned)
        # [Author, 2024]
        cleaned = re.sub(r'\[[\w\s]+,?\s*\d{4}\]', '', cleaned)

    if remove_superscript_numbers:
        # superscript numbers
        cleaned = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+', '', cleaned)
        # (1), (2,3) but not (2024)
        cleaned = re.sub(r'\((\d+(?:,\s*\d+)*)\)(?!\s*\d{3})', '', cleaned)

    # tidy spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
    return cleaned.strip()


def _clean_academic_artifacts(text: str, aggressive: bool = False) -> str:
    """
    Remove DOI, obvious headers/footers, volume/issue lines, etc.
    """
    lines = text.split('\n')
    cleaned_lines: List[str] = []

    for line in lines:
        s = line.strip()

        if not s:
            cleaned_lines.append(line)
            continue

        # short header/footer-like lines
        if len(s) < 80:
            if re.search(r'^Vol\.:?\(?\d+\)', s):
                continue
            if re.search(r'^https?://doi\.org/', s):
                continue
            if re.match(r'^©\s*\d{4}', s):
                continue
            # project-specific journal names
            if s in ("Discover Sustainability", "Research"):
                continue

        # inline DOI
        line = re.sub(r'https?://doi\.org/[\S]+', '', line)
        # inline Vol.
        line = re.sub(r'Vol\.:?\(?\d+\)', '', line)

        if aggressive:
            line = re.sub(r'©\s*The\s+Author\(s\)\s*\d{4}', '', line)

        cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines)
    # collapse multiple blank lines
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    return result


# =========================================================
# 2. pypdf → Docling datamodel fallback
# =========================================================
def _extract_with_docling_datamodel_fallback(
    pdf_path: str,
    *,
    clean_cites: bool = False,
    clean_artifacts: bool = False,
    aggressive_clean: bool = False,
):
    """
    Fallback that mimics a Docling ConvertedDocument using pypdf + docling.datamodel.*
    so the rest of the pipeline still gets a doc-like structure.

    Returns:
        doc (ConvertedDocument-like)
    """
    if _PYPDF_READER is None:
        raise RuntimeError("pypdf not available for docling-datamodel fallback")

    # import here to avoid hard dep if docling isn't installed
    from docling.datamodel.document import (
        ConvertedDocument,
        Page,
        InputDocument,
    )
    from docling.datamodel.base_models import (
        BoundingBox,
        Cluster,
        CoordOrigin,
        TextElement,
    )
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    reader = _PYPDF_READER(pdf_path)
    pdf_path_obj = Path(pdf_path)

    input_doc = InputDocument(
        path_or_stream=pdf_path_obj,
        filename=pdf_path_obj.name,
        pdf_backend=PyPdfiumDocumentBackend,
    )

    doc = ConvertedDocument(input=input_doc)
    doc._source_filename = pdf_path_obj.name

    element_id = 0
    for page_num, pdf_page in enumerate(reader.pages, start=1):
        txt = pdf_page.extract_text() or ""

        if txt.strip():
            if clean_cites:
                txt = _clean_citations(txt)
            if clean_artifacts:
                txt = _clean_academic_artifacts(txt, aggressive=aggressive_clean)

        page = Page(page_no=page_num)

        if txt.strip():
            bbox = BoundingBox(
                l=0.0,
                t=0.0,
                r=1.0,
                b=1.0,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            cluster = Cluster(id=element_id, label="text", bbox=bbox)
            text_elem = TextElement(
                label="text",
                id=element_id,
                page_no=page_num,
                cluster=cluster,
                text=txt,
            )
            if not hasattr(page, '_raw_elements'):
                page._raw_elements = []
            page._raw_elements.append(text_elem)
            element_id += 1

        doc.pages.append(page)

    # assemble
    assembled = []
    for pg in doc.pages:
        if hasattr(pg, '_raw_elements'):
            assembled.extend(pg._raw_elements)
    doc._assembled_elements = assembled

    return doc


def _export_docling_document_to_text(doc) -> str:
    """
    Export a ConvertedDocument-like object to markdown-ish text
    (## Page N + page text).
    """
    parts: List[str] = []
    if hasattr(doc, "_assembled_elements"):
        for elem in doc._assembled_elements:
            if getattr(elem, "text", None):
                parts.append(f"## Page {elem.page_no}\n\n{elem.text}\n")
    return "\n".join(parts).strip() if parts else ""


# =========================================================
# 3. pypdf + cleaning (string) stage
# =========================================================
def _extract_with_pypdf_and_clean(
    pdf_path: str,
    *,
    clean_cites: bool = True,
    clean_artifacts: bool = True,
    aggressive_clean: bool = False,
    return_doc: bool = False,
):
    """
    Lightweight stage: pypdf → per-page text → optional cleaning → (optionally) page map.
    """
    if _PYPDF_READER is None:
        raise RuntimeError("pypdf not available")

    reader = _PYPDF_READER(pdf_path)
    pages_out: List[Dict[str, Any]] = []
    parts: List[str] = []

    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        orig_txt = txt

        if txt.strip():
            if clean_cites:
                txt = _clean_citations(txt)
            if clean_artifacts:
                txt = _clean_academic_artifacts(txt, aggressive=aggressive_clean)

        pages_out.append(
            {
                "page_no": i,
                "original": orig_txt,
                "cleaned": txt,
            }
        )

        if txt.strip():
            parts.append(f"## Page {i}\n\n{txt}\n")

    text = "\n".join(parts).strip()

    if return_doc:
        return {
            "text": text,
            "pages": pages_out,
            "source": "pypdf_clean",
        }
    return text


# =========================================================
# 4. main public entry
# =========================================================
def extract_text_from_pdf_robust(
    path: str,
    *,
    clean_cites: bool = False,
    clean_artifacts: bool = False,
    aggressive_clean: bool = False,
    return_doc: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Docling-first multistage PDF extractor.

    Returns:
        - str (default)
        - or dict with keys: text, pages, source  (if return_doc=True)
    """
    p = Path(path)
    print(f"📄 Extracting text from: {p}")
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # sanity: is it PDF?
    with p.open("rb") as fh:
        head = fh.read(5)
        if head not in (b"%PDF-",):
            raise RuntimeError("Not a PDF (magic header mismatch)")

    # maybe it's actually text/plain but named .pdf – we can check mime
    mtype, _ = mimetypes.guess_type(str(p))
    if str(p).lower().endswith(".txt") or (mtype and mtype.startswith("text/")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if return_doc:
            return {"text": txt, "pages": [], "source": "text"}
        return txt

    # -----------------------------------------------------
    # 1) True Docling
    # -----------------------------------------------------
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            do_figure_extraction=False,
        )
        pipeline_options.table_structure_options.do_cell_matching = False

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        res = converter.convert(str(p))
        doc = res.document

        try:
            text = doc.export_to_text()
        except Exception:
            # fallback assembling from sections
            blocks: List[str] = []
            for sec in getattr(doc, "sections", []) or []:
                if getattr(sec, "title", None):
                    blocks.append(str(sec.title).strip())
                for para in getattr(sec, "paragraphs", []) or []:
                    if getattr(para, "text", None):
                        blocks.append(str(para.text).strip())
            text = "\n\n".join(t for t in blocks if t)

        if text and text.strip():
            if clean_cites:
                text = _clean_citations(text)
            if clean_artifacts:
                text = _clean_academic_artifacts(text, aggressive=aggressive_clean)
            if return_doc:
                return {"text": text.strip(), "pages": [], "source": "docling"}
            return text.strip()
        else:
            print("ℹ️ Docling produced empty text; trying docling-datamodel fallback…")
    except Exception as e:
        print("[Docling] ⚠️ import/convert failed; trying docling-datamodel fallback…")
        # continue to stage 2

    # -----------------------------------------------------
    # 2) pypdf → Docling datamodel (synthetic)
    # -----------------------------------------------------
    try:
        doc2 = _extract_with_docling_datamodel_fallback(
            str(p),
            clean_cites=clean_cites,
            clean_artifacts=clean_artifacts,
            aggressive_clean=aggressive_clean,
        )
        text2 = _export_docling_document_to_text(doc2)
        if text2.strip():
            print(f"✅ docling-datamodel fallback extracted {len(text2)} chars")
            if return_doc:
                # build page map
                pages: List[Dict[str, Any]] = []
                for pg in doc2.pages:
                    pg_txt = ""
                    if hasattr(pg, "_raw_elements"):
                        for el in pg._raw_elements:
                            if getattr(el, "text", None):
                                pg_txt += el.text + "\n"
                    pages.append(
                        {
                            "page_no": pg.page_no,
                            "original": pg_txt.strip(),
                            "cleaned": pg_txt.strip(),
                        }
                    )
                return {
                    "text": text2,
                    "pages": pages,
                    "source": "docling_pypdf",
                }
            return text2
        else:
            print("ℹ️ docling-datamodel fallback empty; trying pypdf-clean…")
    except Exception as e:
        print(f"[docling-datamodel fallback] ⚠️ failed: {e}; trying pypdf-clean…")

    # -----------------------------------------------------
    # 3) pypdf + cleaning (string)
    # -----------------------------------------------------
    try:
        pypdf_res = _extract_with_pypdf_and_clean(
            str(p),
            clean_cites=clean_cites or True,
            clean_artifacts=clean_artifacts or True,
            aggressive_clean=aggressive_clean,
            return_doc=return_doc,
        )
        if return_doc:
            if pypdf_res.get("text", "").strip():
                return pypdf_res
        else:
            if pypdf_res.strip():
                return pypdf_res
        print("ℹ️ pypdf-clean empty; trying PyPDF2…")
    except Exception as e:
        print(f"[pypdf-clean] ⚠️ {e}; trying PyPDF2…")

    # -----------------------------------------------------
    # 4) PyPDF2 (last resort)
    # -----------------------------------------------------
    try:
        import PyPDF2

        text = ""
        pages_extracted = 0
        with p.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh, strict=False)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    pass
            total = len(reader.pages)
            for i, page in enumerate(reader.pages, 1):
                try:
                    t = page.extract_text() or ""
                    if t.strip():
                        if clean_cites:
                            t = _clean_citations(t)
                        if clean_artifacts:
                            t = _clean_academic_artifacts(t, aggressive=aggressive_clean)
                        text += t + "\n\n"
                        pages_extracted += 1
                except Exception as pe:
                    print(f"  ⚠️ PyPDF2 page {i}/{total}: {pe}")

        if text.strip():
            if return_doc:
                return {
                    "text": text.strip(),
                    "pages": [
                        {"page_no": i + 1, "original": None, "cleaned": None}
                        for i in range(pages_extracted)
                    ],
                    "source": "pypdf2",
                }
            return text.strip()
        raise RuntimeError("PyPDF2 returned empty text")
    except Exception as e2:
        print(f"❌ PyPDF2 fallback failed: {e2}")
        raise RuntimeError(
            "PDF extraction failed via: docling → docling-datamodel → pypdf-clean → PyPDF2"
        )


# ---------------------------------------------------------
# CLI / demo (optional)
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python robust_pdf_extractor.py <pdf-path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    doc = extract_text_from_pdf_robust(
        pdf_path,
        clean_cites=True,
        clean_artifacts=True,
        return_doc=True,
    )
    print(f"Source: {doc.get('source')}")
    print(f"Text preview:\n{doc.get('text', '')[:800]}")
    print(f"Pages: {len(doc.get('pages', []))}")
