"""
Knowledge-Base PDF Extraction Pipeline for Government Scheme Documents.

This script extracts and cleans text from government scheme PDFs using PyMuPDF (fitz)
with an integrated OCR fallback (RapidOCR) for scanned or image-based pages.
It preserves document hierarchy, headings, lists, tables, and page markers, while removing
non-printable control characters. It also checks for and reports extraction anomalies
such as empty pages or scanned images without embedded text.

Usage:
    # Process all PDFs in default directory (data/source_pdfs -> data/processed)
    py src/process_pdfs.py

    # Process custom directories
    py src/process_pdfs.py --input-dir data/source_pdfs --output-dir data/processed

    # Process a single PDF with automatic OCR fallback
    py src/process_pdfs.py --file data/source_pdfs/PMVISH.pdf

    # Force OCR on all pages
    py src/process_pdfs.py --file data/source_pdfs/PMVISH.pdf --force-ocr
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pymupdf


# Ensure UTF-8 output encoding in Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Global OCR engine instance (lazy-loaded)
_OCR_ENGINE = None


def get_ocr_engine():
    """Lazy initialize RapidOCR engine."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except ImportError:
            _OCR_ENGINE = None
    return _OCR_ENGINE


def clean_page_text(raw_text: str) -> str:
    """
    Clean extracted text while strictly preserving content, structure,
    headings, bullet points, and numbered lists.
    
    Cleaning steps:
    1. Replace non-breaking spaces (\xa0) with standard space.
    2. Remove zero-width spaces (\u200b) and BOM (\ufeff) often injected by PDF renderers.
    3. Strip trailing whitespace from each line.
    4. Collapse excessive consecutive blank lines (3+ into 2) to maintain clean paragraphs.
    """
    if not raw_text:
        return ""

    # Replace non-breaking spaces with standard space
    text = raw_text.replace("\xa0", " ")

    # Remove zero-width formatting characters that break word searching
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.splitlines()]

    # Rejoin lines
    text = "\n".join(lines)

    # Collapse runs of 3 or more newlines down to 2 newlines (standard paragraph spacing)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def ocr_page_image(
    page: pymupdf.Page, dpi: int = 300, min_confidence: float = 0.40
) -> Tuple[str, List[str], bool]:
    """
    Render a PDF page to a high-resolution image and run OCR.
    Preserves headings, paragraphs, and list layout by grouping
    bounding boxes by line coordinates. Marks low-confidence text as [UNCLEAR].

    Returns:
        (reconstructed_text, unclear_items, ocr_executed_successfully)
    """
    engine = get_ocr_engine()
    if engine is None:
        return "", ["RapidOCR engine is not installed."], False

    # Render page to pixmap
    pix = page.get_pixmap(dpi=dpi)

    # Verify if page image has any ink / non-white pixels
    samples = pix.samples
    # Check if page is entirely white (all RGB components == 255)
    has_ink = any(b < 250 for b in samples[::50])
    if not has_ink:
        # 100% white blank page
        return "", [], True

    img_bytes = pix.tobytes("png")
    try:
        result, _ = engine(img_bytes)
    except Exception as e:
        return "", [f"OCR execution failed: {e}"], False

    if not result:
        return "", [], True

    # Process bounding boxes
    boxes = []
    unclear_items = []
    for item in result:
        coords, txt, score = item[0], item[1].strip(), float(item[2])
        if not txt:
            continue
        y_min = min(pt[1] for pt in coords)
        y_max = max(pt[1] for pt in coords)
        x_min = min(pt[0] for pt in coords)

        if score < min_confidence:
            unclear_items.append(f"Low confidence ({score:.2f}) for text: '{txt}'")
            txt = "[UNCLEAR]"

        boxes.append((y_min, y_max, x_min, txt))

    if not boxes:
        return "", unclear_items, True

    # Sort boxes top-to-bottom, then left-to-right
    boxes.sort(key=lambda b: (b[0], b[2]))

    # Group boxes into lines
    lines = []
    current_line = []
    current_y = None
    line_threshold = (boxes[0][1] - boxes[0][0]) * 0.5 if boxes else 15

    for b in boxes:
        if current_y is None:
            current_y = b[0]
            current_line.append(b)
        elif abs(b[0] - current_y) <= line_threshold:
            current_line.append(b)
        else:
            current_line.sort(key=lambda x: x[2])
            lines.append(" ".join(x[3] for x in current_line))
            current_line = [b]
            current_y = b[0]
            line_threshold = (b[1] - b[0]) * 0.5

    if current_line:
        current_line.sort(key=lambda x: x[2])
        lines.append(" ".join(x[3] for x in current_line))

    ocr_text = clean_page_text("\n".join(lines))
    return ocr_text, unclear_items, True


def extract_pdf(pdf_path: Path, force_ocr: bool = False) -> Dict:
    """
    Extract text and metadata from a single PDF file using PyMuPDF and OCR fallback.

    Returns:
        dict containing:
            - filename: str
            - total_pages: int
            - total_characters: int
            - total_words: int
            - page_data: list of dicts with page details
            - issues: list of issue descriptions
            - full_content: formatted string with page markers
            - ocr_used: bool
    """
    issues = []
    page_data = []
    pages_output = []
    total_characters = 0
    total_words = 0
    ocr_used = False

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        return {
            "filename": pdf_path.name,
            "total_pages": 0,
            "total_characters": 0,
            "total_words": 0,
            "page_data": [],
            "issues": [f"Failed to open PDF: {e}"],
            "full_content": "",
            "ocr_used": False,
        }

    total_pages = len(doc)
    if total_pages == 0:
        issues.append("Document has 0 pages.")

    for page_idx, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        images = page.get_images()
        cleaned_text = clean_page_text(raw_text)

        page_used_ocr = False
        unclear_list = []

        # If text extraction yielded nothing or force_ocr is requested, run OCR
        if force_ocr or len(cleaned_text) == 0:
            ocr_text, unclear_list, ocr_success = ocr_page_image(page)
            page_used_ocr = True
            ocr_used = True

            if ocr_text:
                cleaned_text = ocr_text
                if unclear_list:
                    issues.append(
                        f"Page {page_idx}: {len(unclear_list)} segment(s) marked [UNCLEAR] due to low confidence."
                    )
            elif not ocr_success:
                issues.append(f"Page {page_idx}: OCR failed: {'; '.join(unclear_list)}")

        page_chars = len(cleaned_text)
        page_words = len(cleaned_text.split())

        if page_chars == 0:
            if page_used_ocr:
                issues.append(
                    f"Page {page_idx}: OCR executed successfully. No readable text found (page is visually blank)."
                )
                page_body = "[No readable text found on this page - page is blank]"
            elif len(images) > 0:
                issues.append(
                    f"Page {page_idx}: No embedded text found, but {len(images)} image(s) detected (may require OCR)."
                )
                page_body = "[No extractable text found on this page - contains images only]"
            else:
                issues.append(f"Page {page_idx}: Completely blank (0 words, no images).")
                page_body = "[Blank page / No extractable text found]"
        else:
            page_body = cleaned_text

        page_data.append({
            "page_number": page_idx,
            "characters": page_chars,
            "words": page_words,
            "images_count": len(images),
            "used_ocr": page_used_ocr,
            "unclear_count": len(unclear_list),
            "text": page_body,
        })

        total_characters += page_chars
        total_words += page_words

        # Page demarcation
        pages_output.append(f"--- Page {page_idx} ---\n{page_body}")

    doc.close()

    full_content = "\n\n".join(pages_output)

    return {
        "filename": pdf_path.name,
        "total_pages": total_pages,
        "total_characters": total_characters,
        "total_words": total_words,
        "page_data": page_data,
        "issues": issues,
        "full_content": full_content,
        "ocr_used": ocr_used,
    }


def process_pdf_file(pdf_path: Path, output_dir: Path, force_ocr: bool = False) -> Dict:
    """
    Extract text from a PDF and save the cleaned TXT file.
    """
    result = extract_pdf(pdf_path, force_ocr=force_ocr)

    # Prepare output path
    output_dir.mkdir(parents=True, exist_ok=True)
    out_filename = pdf_path.stem + ".txt"
    out_path = output_dir / out_filename

    # Save to file with UTF-8 encoding
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result["full_content"])

    result["output_file"] = str(out_path)
    return result


def process_all_pdfs(
    input_dir: Path,
    output_dir: Path,
    single_file: Optional[Path] = None,
    force_ocr: bool = False,
) -> List[Dict]:
    """
    Process all PDFs in input_dir or a single designated file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if single_file:
        if not single_file.exists():
            raise FileNotFoundError(f"Specified file not found: {single_file}")
        pdf_files = [single_file]
    else:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        pdf_files = sorted(
            [p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf"],
            key=lambda x: x.name.lower(),
        )

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return []

    results = []
    for pdf_path in pdf_files:
        res = process_pdf_file(pdf_path, output_dir, force_ocr=force_ocr)
        results.append(res)

    return results


def print_summary_report(results: List[Dict]) -> None:
    """
    Print an inspection and extraction summary report.
    """
    print("\n" + "=" * 90)
    print("                GOV-19 KNOWLEDGE-BASE PDF EXTRACTION REPORT")
    print("=" * 90)
    print(
        f"{'Filename':<16} | {'Pages':<6} | {'Words':<8} | {'Characters':<11} | {'Status / Issues':<40}"
    )
    print("-" * 90)

    total_pages_all = 0
    total_words_all = 0
    total_chars_all = 0

    for r in results:
        fname = r["filename"]
        pages = r["total_pages"]
        words = r["total_words"]
        chars = r["total_characters"]
        issues = r["issues"]

        total_pages_all += pages
        total_words_all += words
        total_chars_all += chars

        if not issues:
            status = "Clean extraction (OK)"
        else:
            status = "; ".join(issues)

        # Truncate status if too long for one line in table
        if len(status) > 40:
            status_line = status[:37] + "..."
        else:
            status_line = status

        print(f"{fname:<16} | {pages:<6} | {words:<8} | {chars:<11} | {status_line:<40}")

    print("-" * 90)
    print(
        f"{'TOTAL (' + str(len(results)) + ' files)':<16} | {total_pages_all:<6} | {total_words_all:<8} | {total_chars_all:<11} |"
    )
    print("=" * 90)

    # Detailed issue breakdown if any
    any_issues = any(r["issues"] for r in results)
    if any_issues:
        print("\nDETAILED EXTRACTION ISSUES & FINDINGS:")
        for r in results:
            if r["issues"]:
                print(f"  • {r['filename']}:")
                for iss in r["issues"]:
                    print(f"      - {iss}")
    else:
        print("\nNo extraction issues detected across documents.")

    print("\nPROCESSED OUTPUT FILES:")
    for r in results:
        if "output_file" in r:
            print(f"  • {r['output_file']}")
    print("=" * 90 + "\n")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    default_input = base_dir / "data" / "source_pdfs"
    default_output = base_dir / "data" / "processed"

    parser = argparse.ArgumentParser(
        description="Extract and clean text from Government Scheme PDFs for GOV-19 Knowledge Base."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=default_input,
        help=f"Directory containing source PDFs (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=default_output,
        help=f"Directory to save processed TXT files (default: {default_output})",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Process a single PDF file instead of entire directory",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR execution on every page of the document",
    )

    args = parser.parse_args()

    results = process_all_pdfs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        single_file=args.file,
        force_ocr=args.force_ocr,
    )
    print_summary_report(results)


if __name__ == "__main__":
    main()
