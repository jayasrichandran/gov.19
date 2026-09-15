"""
Document Chunking Pipeline for GOV-19 Knowledge Base.

Reads the 4 verified government scheme text files from data/processed/:
- AYUSHHWC.txt (AYUSH Health and Wellness Centre)
- PMFBY.txt (Pradhan Mantri Fasal Bima Yojana)
- PMSVAN.txt (PM SVANidhi)
- PMUY2.0.txt (PM Ujjwala Yojana 2.0)

Splits text into semantically cohesive, page-tracked chunks preserving:
- scheme name
- source document (PDF filename)
- page number
- chunk text and metadata
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure UTF-8 output encoding in Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


SCHEME_CONFIG = {
    "AYUSHHWC.txt": {
        "scheme": "AYUSH Health and Wellness Centre - Ayushman Bharat Yojana",
        "source_document": "AYUSHHWC.pdf",
        "prefix": "AYUSH",
    },
    "PMFBY.txt": {
        "scheme": "PMFBY",
        "source_document": "PMFBY.pdf",
        "prefix": "PMFBY",
    },
    "PMSVAN.txt": {
        "scheme": "PM SVANidhi",
        "source_document": "PMSVAN.pdf",
        "prefix": "PMSVAN",
    },
    "PMUY2.0.txt": {
        "scheme": "PMUY 2.0",
        "source_document": "PMUY2.0.pdf",
        "prefix": "PMUY",
    },
}


def parse_pages_from_text(file_content: str) -> List[Dict]:
    """
    Parse a processed TXT file into separate pages based on '--- Page {n} ---' markers.
    """
    page_pattern = re.compile(r"^---\s*Page\s+(\d+)\s*---", re.MULTILINE)
    matches = list(page_pattern.finditer(file_content))

    pages = []
    if not matches:
        # Fallback if no page markers
        pages.append({"page_number": 1, "text": file_content.strip()})
        return pages

    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(file_content)
        page_text = file_content[start_idx:end_idx].strip()
        if page_text:
            pages.append({"page_number": page_num, "text": page_text})

    return pages


def split_text_into_chunks(
    page_text: str,
    scheme_name: str,
    source_doc: str,
    page_num: int,
    prefix: str,
    chunk_index_start: int,
    max_chunk_chars: int = 600,
    chunk_overlap_chars: int = 100,
) -> List[Dict]:
    """
    Split page text into cohesive chunks, respecting section headers,
    paragraphs, and list items.
    """
    chunks = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]

    current_chunk = []
    current_length = 0
    chunk_counter = chunk_index_start

    def flush_chunk(accumulated_lines: List[str]) -> Optional[Dict]:
        nonlocal chunk_counter
        chunk_body = "\n".join(accumulated_lines).strip()
        if not chunk_body:
            return None
        chunk_id = f"{prefix}_p{page_num}_c{chunk_counter:02d}"
        chunk_counter += 1
        return {
            "chunk_id": chunk_id,
            "scheme": scheme_name,
            "source_document": source_doc,
            "page_number": page_num,
            "text": chunk_body,
            "char_count": len(chunk_body),
            "word_count": len(chunk_body.split()),
        }

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph is larger than max_chunk_chars, split on lines or sentences
        if para_len > max_chunk_chars:
            lines = [l.strip() for l in para.splitlines() if l.strip()]
            for line in lines:
                if current_length + len(line) + 1 > max_chunk_chars and current_chunk:
                    chk = flush_chunk(current_chunk)
                    if chk:
                        chunks.append(chk)
                    # Overlap with last line if suitable
                    if chunk_overlap_chars > 0 and current_chunk:
                        last_line = current_chunk[-1]
                        current_chunk = [last_line] if len(last_line) <= chunk_overlap_chars else []
                        current_length = sum(len(l) for l in current_chunk) + len(current_chunk)
                    else:
                        current_chunk = []
                        current_length = 0

                current_chunk.append(line)
                current_length += len(line) + 1
        else:
            if current_length + para_len + 2 > max_chunk_chars and current_chunk:
                chk = flush_chunk(current_chunk)
                if chk:
                    chunks.append(chk)
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_len + 2

    if current_chunk:
        chk = flush_chunk(current_chunk)
        if chk:
            chunks.append(chk)

    return chunks


def process_and_chunk_documents(
    input_dir: Path, output_file: Path
) -> List[Dict]:
    """
    Read all 4 target scheme files and write combined chunks JSON.
    """
    all_chunks = []
    print("\n" + "=" * 70)
    print("                GOV-19 DOCUMENT CHUNKING PIPELINE")
    print("=" * 70)

    for filename, cfg in SCHEME_CONFIG.items():
        file_path = input_dir / filename
        if not file_path.exists():
            print(f"Warning: Expected processed file not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        pages = parse_pages_from_text(content)
        doc_chunks = []
        chunk_idx = 1

        for p in pages:
            page_chunks = split_text_into_chunks(
                page_text=p["text"],
                scheme_name=cfg["scheme"],
                source_doc=cfg["source_document"],
                page_num=p["page_number"],
                prefix=cfg["prefix"],
                chunk_index_start=chunk_idx,
            )
            chunk_idx += len(page_chunks)
            doc_chunks.extend(page_chunks)

        all_chunks.extend(doc_chunks)
        print(
            f"• {filename:<14} | Scheme: {cfg['scheme']:<20} | Pages: {len(pages)} | Chunks: {len(doc_chunks)}"
        )

    print("-" * 70)
    print(f"Total Chunks Generated Across 4 Schemes: {len(all_chunks)}")
    print("=" * 70)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunks saved to: {output_file}\n")
    return all_chunks


def main():
    base_dir = Path(__file__).resolve().parent.parent
    default_input = base_dir / "data" / "processed"
    default_output = base_dir / "data" / "processed" / "chunks.json"

    parser = argparse.ArgumentParser(description="Chunk GOV-19 processed scheme documents.")
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=default_input,
        help=f"Directory containing processed TXT files (default: {default_input})",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=default_output,
        help=f"Path to save chunks JSON (default: {default_output})",
    )

    args = parser.parse_args()
    process_and_chunk_documents(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
