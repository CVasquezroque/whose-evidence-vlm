#!/usr/bin/env python
"""Write an explicitly labeled synthetic illustration under work_root/examples."""
from __future__ import annotations
from evidence_ownership.rendering.synthetic import make_synthetic_scene
from evidence_ownership.rendering.factorial import render_scene
from evidence_ownership.rendering.qc import contact_sheet
from evidence_ownership.utils.config import get_paths
from evidence_ownership.utils.io import write_json


def main() -> None:
    output = get_paths().work_root / 'examples' / 'synthetic'
    output.mkdir(parents=True, exist_ok=True)
    image, item = make_synthetic_scene()
    source = output / 'source.png'
    image.save(source, format='PNG', optimize=False)
    write_json(output / 'item.json', item)
    record = render_scene(item, source, output / 'rendered')
    contact_sheet(output / 'rendered', output / 'factorial_schematic.png', schematic=True)
    print(f'{record["status"]}: {output / "factorial_schematic.png"}')
    if record['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
