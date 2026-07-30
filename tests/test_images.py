"""Tests for openkb.images — base64 extraction and relative image copy."""

from __future__ import annotations

import base64

import pymupdf

from openkb.images import (
    convert_pdf_to_pages,
    convert_pdf_with_images,
    copy_relative_images,
    extract_base64_images,
    extract_pdf_images,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8  # minimal fake PNG bytes
FAKE_JPG = b"\xff\xd8\xff" + b"\x00" * 8  # minimal fake JPEG bytes


# ---------------------------------------------------------------------------
# extract_base64_images
# ---------------------------------------------------------------------------


class TestExtractBase64Images:
    def test_no_images_returns_unchanged(self, tmp_path):
        md = "# Hello\n\nSome text without any images."
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)
        result = extract_base64_images(md, "doc", images_dir)
        assert result == md

    def test_single_base64_image_extracted(self, tmp_path):
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)
        b64 = _make_b64(FAKE_PNG)
        md = f"![alt text](data:image/png;base64,{b64})"
        result = extract_base64_images(md, "doc", images_dir)

        # Result should reference a saved file, not the raw base64
        assert "data:image/png;base64," not in result
        assert "![alt text](images/doc/img_001.png)" == result

        # File should exist on disk
        saved = images_dir / "img_001.png"
        assert saved.exists()
        assert saved.read_bytes() == FAKE_PNG

    def test_multiple_base64_images_numbered_sequentially(self, tmp_path):
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)
        b64_png = _make_b64(FAKE_PNG)
        b64_jpg = _make_b64(FAKE_JPG)
        md = f"![fig1](data:image/png;base64,{b64_png})\n![fig2](data:image/jpeg;base64,{b64_jpg})"
        result = extract_base64_images(md, "doc", images_dir)

        assert "![fig1](images/doc/img_001.png)" in result
        assert "![fig2](images/doc/img_002.jpeg)" in result
        assert (images_dir / "img_001.png").exists()
        assert (images_dir / "img_002.jpeg").exists()

    def test_invalid_base64_leaves_original(self, tmp_path, caplog):
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)
        bad = "NOT_VALID_BASE64!!!"
        md = f"![alt](data:image/png;base64,{bad})"
        import logging

        with caplog.at_level(logging.WARNING, logger="openkb.images"):
            result = extract_base64_images(md, "doc", images_dir)
        assert result == md  # unchanged
        # No files created
        assert list(images_dir.iterdir()) == []

    def test_mixed_valid_invalid_base64(self, tmp_path, caplog):
        """Valid image extracted; invalid image left in place."""
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)
        b64 = _make_b64(FAKE_PNG)
        bad = "BADBAD!!!"
        md = f"![good](data:image/png;base64,{b64})\n![bad](data:image/png;base64,{bad})"
        import logging

        with caplog.at_level(logging.WARNING, logger="openkb.images"):
            result = extract_base64_images(md, "doc", images_dir)
        assert "![good](images/doc/img_001.png)" in result
        assert f"data:image/png;base64,{bad}" in result


# ---------------------------------------------------------------------------
# copy_relative_images
# ---------------------------------------------------------------------------


class TestCopyRelativeImages:
    def test_existing_relative_image_copied_and_rewritten(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        img_file = source_dir / "diagram.png"
        img_file.write_bytes(FAKE_PNG)

        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)

        md = "![diagram](diagram.png)"
        result = copy_relative_images(md, source_dir, "doc", images_dir)

        assert "![diagram](images/doc/diagram.png)" == result
        assert (images_dir / "diagram.png").read_bytes() == FAKE_PNG

    def test_missing_relative_image_leaves_original(self, tmp_path, caplog):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)

        md = "![missing](missing.png)"
        import logging

        with caplog.at_level(logging.WARNING, logger="openkb.images"):
            result = copy_relative_images(md, source_dir, "doc", images_dir)
        assert result == md  # unchanged
        assert list(images_dir.iterdir()) == []

    def test_http_url_not_processed(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)

        md = "![logo](https://example.com/logo.png)"
        result = copy_relative_images(md, source_dir, "doc", images_dir)
        assert result == md  # HTTP URLs left untouched

    def test_data_url_not_processed(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)

        b64 = _make_b64(FAKE_PNG)
        md = f"![img](data:image/png;base64,{b64})"
        result = copy_relative_images(md, source_dir, "doc", images_dir)
        assert result == md  # data URIs left untouched

    def test_multiple_relative_images_all_copied(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "a.png").write_bytes(FAKE_PNG)
        (source_dir / "b.jpg").write_bytes(FAKE_JPG)

        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)

        md = "![a](a.png)\n![b](b.jpg)"
        result = copy_relative_images(md, source_dir, "doc", images_dir)

        assert "![a](images/doc/a.png)" in result
        assert "![b](images/doc/b.jpg)" in result
        assert (images_dir / "a.png").exists()
        assert (images_dir / "b.jpg").exists()

    def test_same_basename_different_dirs_no_overwrite(self, tmp_path):
        # Two distinct images sharing a basename must not overwrite each other
        # (which would lose one image and point both links at the survivor).
        source_dir = tmp_path / "source"
        (source_dir / "a").mkdir(parents=True)
        (source_dir / "b").mkdir(parents=True)
        (source_dir / "a" / "logo.png").write_bytes(FAKE_PNG)
        (source_dir / "b" / "logo.png").write_bytes(FAKE_JPG)

        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)

        md = "![a](a/logo.png)\n![b](b/logo.png)"
        result = copy_relative_images(md, source_dir, "doc", images_dir)

        saved = sorted(p.name for p in images_dir.iterdir())
        assert len(saved) == 2  # both copied, neither overwritten
        assert {(images_dir / n).read_bytes() for n in saved} == {FAKE_PNG, FAKE_JPG}
        links = sorted(line.split("](")[1].rstrip(")") for line in result.strip().splitlines())
        assert links[0] != links[1]  # links point at different files

    def test_same_image_referenced_twice_is_copied_once(self, tmp_path):
        # Identical source referenced twice: copy once, both links agree.
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "logo.png").write_bytes(FAKE_PNG)
        images_dir = tmp_path / "images" / "doc"
        images_dir.mkdir(parents=True)

        md = "![x](logo.png)\n![y](logo.png)"
        result = copy_relative_images(md, source_dir, "doc", images_dir)

        assert [p.name for p in images_dir.iterdir()] == ["logo.png"]
        assert result.count("images/doc/logo.png") == 2


# ---------------------------------------------------------------------------
# Note-relative resolution (Obsidian / GitHub compatibility)
# ---------------------------------------------------------------------------


class TestNoteRelativeResolution:
    """Generated ``![...](...)`` links must resolve relative to the note.

    Source pages live at ``wiki/sources/<doc>.md`` and their images at
    ``wiki/sources/images/<doc>/`` — renderers that resolve links relative
    to the containing file (Obsidian, GitHub, VS Code) must find the image.
    A wiki-root-relative link (``sources/images/...``) would resolve to the
    non-existent ``wiki/sources/sources/images/...`` from those notes.
    """

    @staticmethod
    def _link_paths(markdown: str) -> list[str]:
        import re

        return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)

    def test_base64_image_ref_resolves_from_sources_note(self, tmp_path):
        wiki = tmp_path / "wiki"
        note = wiki / "sources" / "doc.md"
        images_dir = wiki / "sources" / "images" / "doc"
        images_dir.mkdir(parents=True)

        md = f"![alt](data:image/png;base64,{_make_b64(FAKE_PNG)})"
        result = extract_base64_images(md, "doc", images_dir)
        note.write_text(result, encoding="utf-8")

        for rel in self._link_paths(result):
            assert (note.parent / rel).exists(), rel

    def test_copied_image_ref_resolves_from_sources_note(self, tmp_path):
        source_dir = tmp_path / "clip"
        source_dir.mkdir()
        (source_dir / "figure.png").write_bytes(FAKE_PNG)

        wiki = tmp_path / "wiki"
        note = wiki / "sources" / "doc.md"
        images_dir = wiki / "sources" / "images" / "doc"
        images_dir.mkdir(parents=True)

        result = copy_relative_images("![f](figure.png)", source_dir, "doc", images_dir)
        note.write_text(result, encoding="utf-8")

        for rel in self._link_paths(result):
            assert (note.parent / rel).exists(), rel


# ---------------------------------------------------------------------------
# Repeated header/footer logo filtering (PDF extractors)
# ---------------------------------------------------------------------------


def _make_pixmap(width: int, height: int, color: tuple[int, int, int]) -> pymupdf.Pixmap:
    samples = bytes(color) * width * height
    return pymupdf.Pixmap(pymupdf.csRGB, width, height, samples, False)


def _make_pdf_with_repeated_logo(path, n_pages: int = 4):
    """A synthetic PDF: the same 60x60 "logo" pixmap on every page (byte-for-byte
    identical, as a real letterhead logo would be), plus a distinct 100x100
    "figure" pixmap on page 2 only. Both are above _MIN_IMAGE_DIM."""
    logo = _make_pixmap(60, 60, (200, 0, 0))
    figure = _make_pixmap(100, 100, (0, 200, 0))

    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page(width=300, height=400)
        page.insert_image(pymupdf.Rect(10, 10, 70, 70), pixmap=logo)
        if i == 1:
            page.insert_image(pymupdf.Rect(10, 100, 110, 200), pixmap=figure)
    doc.save(str(path))
    doc.close()


class TestRepeatedImageFiltering:
    """A logo/watermark repeated byte-for-byte across many pages is a running
    header/footer artifact, not page content — it must be dropped everywhere,
    even though it's well above _MIN_IMAGE_DIM (unlike icons/bullets)."""

    def test_convert_pdf_to_pages_drops_repeated_logo(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        _make_pdf_with_repeated_logo(pdf_path)
        images_dir = tmp_path / "images"

        pages = convert_pdf_to_pages(pdf_path, "doc", images_dir)

        all_images = [img["path"] for p in pages for img in p["images"]]
        assert len(all_images) == 1, all_images
        assert "![image](" not in pages[0]["content"]
        assert "![image](" in pages[1]["content"]
        assert len(list(images_dir.glob("*.png"))) == 1

    def test_convert_pdf_with_images_drops_repeated_logo(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        _make_pdf_with_repeated_logo(pdf_path)
        images_dir = tmp_path / "images"

        markdown = convert_pdf_with_images(pdf_path, "doc", images_dir)

        assert markdown.count("![image](") == 1
        assert len(list(images_dir.glob("*.png"))) == 1

    def test_extract_pdf_images_drops_repeated_logo(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        _make_pdf_with_repeated_logo(pdf_path)
        images_dir = tmp_path / "images"

        page_images = extract_pdf_images(pdf_path, "doc", images_dir)

        all_images = [p for imgs in page_images.values() for p in imgs]
        assert len(all_images) == 1, all_images
        assert list(page_images.keys()) == [2]

    def test_non_repeated_images_are_kept(self, tmp_path):
        """Two distinct large images, each appearing only once, must both survive —
        the filter targets exact-byte repetition, not "any image on multiple pages"."""
        pdf_path = tmp_path / "doc.pdf"
        img_a = _make_pixmap(100, 100, (10, 20, 30))
        img_b = _make_pixmap(100, 100, (40, 50, 60))
        doc = pymupdf.open()
        for i, img in enumerate([img_a, img_b]):
            page = doc.new_page(width=300, height=400)
            page.insert_image(pymupdf.Rect(10, 10, 110, 110), pixmap=img)
        doc.save(str(pdf_path))
        doc.close()
        images_dir = tmp_path / "images"

        pages = convert_pdf_to_pages(pdf_path, "doc", images_dir)

        all_images = [img["path"] for p in pages for img in p["images"]]
        assert len(all_images) == 2, all_images


# ---------------------------------------------------------------------------
# Reading order (image interleaved with text, not dumped at page end)
# ---------------------------------------------------------------------------


def _make_pdf_with_image_between_text(path):
    """A page where the image is drawn LAST in the content stream (as many PDF
    exporters do — a text pass, then an image/annotation pass) but sits
    visually BETWEEN two text blocks. Without ``sort=True`` on
    ``get_text("dict")``, PyMuPDF returns blocks in content-stream order,
    stranding the image after both text blocks regardless of where it
    visually belongs."""
    img = _make_pixmap(80, 80, (200, 0, 0))
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((20, 30), "Step 1: do something first.")
    page.insert_text((20, 250), "Step 3: do something after the image.")
    page.insert_image(pymupdf.Rect(20, 100, 100, 180), pixmap=img)  # inserted last
    doc.save(str(path))
    doc.close()


class TestReadingOrder:
    def test_convert_pdf_with_images_places_image_between_text(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        _make_pdf_with_image_between_text(pdf_path)
        images_dir = tmp_path / "images"

        markdown = convert_pdf_with_images(pdf_path, "doc", images_dir)

        step1 = markdown.index("Step 1")
        image = markdown.index("![image](")
        step3 = markdown.index("Step 3")
        assert step1 < image < step3, markdown

    def test_convert_pdf_to_pages_places_image_between_text(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        _make_pdf_with_image_between_text(pdf_path)
        images_dir = tmp_path / "images"

        pages = convert_pdf_to_pages(pdf_path, "doc", images_dir)
        content = pages[0]["content"]

        step1 = content.index("Step 1")
        image = content.index("![image](")
        step3 = content.index("Step 3")
        assert step1 < image < step3, content
