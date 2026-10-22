"""Contact sheets and HTML galleries for visual inspection of candidate scenes."""
from __future__ import annotations

import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BOX_COLORS = {"A": (255, 200, 0), "B": (0, 200, 255)}


def draw_candidate(img: Image.Image, cand: dict, thickness: int = 3) -> Image.Image:
    """Original image with COCO person boxes labelled by canonical identity A/B (inspection only;
    these are not the experimental markers)."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(14, im.width // 30))
    for p in ("A", "B"):
        x, y, w, h = cand["persons"][p]["bbox_xywh"]
        d.rectangle([x, y, x + w, y + h], outline=BOX_COLORS[p], width=thickness)
        d.rectangle([x, max(0, y - font.size - 4), x + font.size + 6, max(font.size + 4, y)], fill=BOX_COLORS[p])
        d.text((x + 3, max(0, y - font.size - 3)), p, fill=(0, 0, 0), font=font)
    return im


def thumb_with_caption(im: Image.Image, caption: str, size: int) -> Image.Image:
    im = im.copy()
    im.thumbnail((size, size), Image.LANCZOS)
    tile = Image.new("RGB", (size, size + 22), (32, 32, 32))
    tile.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    ImageDraw.Draw(tile).text((4, size + 4), caption, fill=(240, 240, 240), font=ImageFont.load_default(size=13))
    return tile


def contact_sheet(tiles: list[Image.Image], cols: int, title: str) -> Image.Image:
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h + 30), (16, 16, 16))
    ImageDraw.Draw(sheet).text((8, 6), title, fill=(255, 255, 255), font=ImageFont.load_default(size=16))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, 30 + (i // cols) * h))
    return sheet


def gallery_html(title: str, entries: list[dict], note: str = "") -> str:
    """entries: {"img": relative path, "caption": str, "meta": dict}."""
    cards = []
    for e in entries:
        meta = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                       for k, v in e.get("meta", {}).items())
        cards.append(
            f'<figure><a href="{html.escape(e["img"])}" target="_blank"><img loading="lazy" '
            f'src="{html.escape(e["img"])}"></a><figcaption><b>{html.escape(e["caption"])}</b>'
            f"<table>{meta}</table></figcaption></figure>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
figure{{margin:0;background:#1d1d1d;padding:8px;border-radius:6px}}img{{width:100%;height:auto}}
table{{font-size:12px;border-collapse:collapse}}th{{text-align:left;padding-right:8px;color:#aaa}}
.note{{color:#fc6}}</style></head><body><h1>{html.escape(title)}</h1><p class="note">{html.escape(note)}</p>
<div class="grid">{''.join(cards)}</div></body></html>"""
