import pytest

from coursemate.chunking import chunk_document, split_sections, split_text

NOTES = """# Hash tables

Intro paragraph.

## Collisions

Two keys can land in the same slot.

### Chaining

Each slot holds a list.

## Resizing

Double the array.
"""


def test_sections_keep_their_heading_path():
    assert split_sections(NOTES) == [
        ("Hash tables", "Intro paragraph."),
        ("Hash tables > Collisions", "Two keys can land in the same slot."),
        ("Hash tables > Collisions > Chaining", "Each slot holds a list."),
        ("Hash tables > Resizing", "Double the array."),
    ]


def test_text_before_the_first_heading_is_kept():
    assert split_sections("Preface.\n\n# Title\n\nBody.") == [
        ("", "Preface."),
        ("Title", "Body."),
    ]


def test_short_paragraphs_are_packed_together():
    assert split_text("aaaa\n\nbbbb\n\ncccc", max_chars=10, overlap=2) == [
        "aaaa\n\nbbbb",
        "cccc",
    ]


def test_long_paragraph_is_cut_into_overlapping_windows():
    text = "abcdefghij" * 3
    pieces = split_text(text, max_chars=12, overlap=4)

    assert all(len(p) <= 12 for p in pieces)
    assert pieces[0][-4:] == pieces[1][:4]
    assert pieces[0] + "".join(p[4:] for p in pieces[1:]) == text


def test_overlap_must_be_smaller_than_max_chars():
    with pytest.raises(ValueError):
        split_text("anything", max_chars=10, overlap=10)


def test_chunks_are_numbered_per_document():
    chunks = chunk_document("hash-tables.md", NOTES)

    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert {c.source for c in chunks} == {"hash-tables.md"}
    assert chunks[2].heading == "Hash tables > Collisions > Chaining"
