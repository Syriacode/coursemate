"""Split markdown notes into labelled chunks for embedding."""

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    index: int
    text: str


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading path, body) for each non-empty section."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(path), text))
        body.clear()

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2)]
        else:
            body.append(line)
    flush()
    return sections


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs into pieces of at most max_chars; window long paragraphs."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            step = max_chars - overlap
            start = 0
            while True:
                pieces.append(paragraph[start : start + max_chars])
                if start + max_chars >= len(paragraph):
                    break
                start += step
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = paragraph

    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    source: str, markdown: str, max_chars: int = 1000, overlap: int = 150
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in split_sections(markdown):
        for piece in split_text(body, max_chars, overlap):
            chunks.append(Chunk(source, heading, len(chunks), piece))
    return chunks
