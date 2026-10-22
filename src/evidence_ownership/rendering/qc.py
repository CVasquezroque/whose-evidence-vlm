"""Small lossless contact sheets for human rendering review."""
from __future__ import annotations
import html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .masks import load_mask


def contact_sheet(scene_dir: Path, output: Path, *, schematic: bool = False) -> None:
    keys = ['clean', 'I11', 'I01', 'I10', 'I00'] if schematic else ['canvas', 'masks', 'clean', 'I11', 'I01', 'I10', 'I00', 'iso_A', 'iso_B']
    cols = 3 if schematic else 3
    tile, title, banner = 340, 30, 45 if schematic else 0
    rows = (len(keys)+cols-1)//cols
    sheet = Image.new('RGB', (cols*tile, rows*(tile+title)+banner), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    if schematic:
        draw.text((12, 12), 'SYNTHETIC ILLUSTRATION - not COCO data', font=font, fill='black')
    for i, key in enumerate(keys):
        if key == 'masks':
            with Image.open(scene_dir / 'canvas.png') as source:
                array = np.array(source)
            for name, color in zip(('M_A', 'M_B', 'S_A', 'S_B'), ((255, 0, 0), (0, 80, 255), (0, 255, 80), (255, 0, 255))):
                mask = load_mask(scene_dir / f'{name}.png')
                array[mask] = (array[mask]*0.4+np.array(color)*0.6).astype(np.uint8)
            panel = Image.fromarray(array)
        else:
            with Image.open(scene_dir / f'{key}.png') as source:
                panel = source.copy()
        # Schematic puts clean first; factorial panels form the requested 2x2 block.
        if schematic:
            col, row = ((0, 0) if key == 'clean' else (1+(i-1)%2, (i-1)//2))
        else:
            col, row = i % cols, i // cols
        x, y = col*tile, banner+row*(tile+title)
        draw.text((x+6, y+5), key if key != 'masks' else 'M_A red / M_B blue / S_A green / S_B pink', font=ImageFont.load_default(size=12) if key == 'masks' else font, fill='black')
        sheet.paste(panel.resize((tile, tile), Image.Resampling.LANCZOS), (x, y+title))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format='PNG', optimize=False)


def write_gallery(qc_dir: Path, records: list[dict]) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for record in records:
        scene = html.escape(record['scene_id'], quote=True)
        entries.append(f'<li>{scene}: {record["status"]} <a href="{scene}.png"><img width="300" src="{scene}.png" alt="{scene}"></a></li>')
    (qc_dir / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Rendering QC</title><h1>Rendering QC</h1><ul>'+''.join(entries)+'</ul></html>\n', encoding='utf-8')
