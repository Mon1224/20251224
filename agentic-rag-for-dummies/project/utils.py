import os
import shutil
import config
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import glob
import tiktoken
from functools import lru_cache


def clear_directory_contents(directory: Path) -> None:
    """Delete everything under directory but not the directory itself (safe for Docker volume / bind mount roots)."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _markdown_name(pdf_path) -> str:
    """Return the markdown file name for a PDF, replacing only the final
    ``.pdf`` suffix.

    Do NOT use ``Path.with_suffix(".md")`` here: for file names containing
    multiple dot-separated segments (e.g. ``V1.5.pdf(320893)_TMP.pdf``) it
    replaces the wrong segment and writes the output under a different name
    than the one the caller expects, which makes uploads fail silently.
    """
    return os.path.splitext(os.path.basename(str(pdf_path)))[0] + ".md"


def pdf_to_markdown(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False, image_path=None)
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / _markdown_name(pdf_path)
    output_path.write_bytes(md_cleaned.encode('utf-8'))

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = Path(output_dir) / _markdown_name(pdf_path)
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)

@lru_cache(maxsize=1)
def _get_token_encoding():
    try:
        return tiktoken.encoding_for_model("gpt-4")
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def estimate_context_tokens(messages: list) -> int:
    contents = [
        str(msg.content)
        for msg in messages
        if hasattr(msg, "content") and msg.content
    ]
    encoding = _get_token_encoding()
    if encoding is None:
        return sum(max(1, len(content) // 4) for content in contents)
    return sum(len(encoding.encode(content)) for content in contents)
