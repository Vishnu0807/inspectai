"""Create a clearly synthetic bottle dataset; no real defect claims."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def bottle(seed: int, defect: str | None = None) -> Image.Image:
    rng = np.random.default_rng(seed)
    backdrop = int(rng.integers(230, 239))
    image = Image.new('RGB', (256, 256), (backdrop, backdrop, backdrop - 3))
    d = ImageDraw.Draw(image)
    x = int(rng.integers(-4, 5))
    y = int(rng.integers(-3, 4))
    body = (x + 81, y + 58, x + 175, y + 218)
    shadow = (x + 89, y + 208, x + 189, y + 231)
    d.ellipse(shadow, fill=(205, 208, 202))
    d.rounded_rectangle(body, radius=19, fill=(76, 139, 143), outline=(45, 89, 91), width=3)
    d.rounded_rectangle((x + 104, y + 42, x + 152, y + 69), radius=4, fill=(62, 116, 119))
    d.rounded_rectangle((x + 99, y + 31, x + 157, y + 48), radius=3, fill=(48, 72, 74))
    d.rectangle((x + 111, y + 30, x + 145, y + 35), fill=(86, 103, 104))
    d.rounded_rectangle((x + 90, y + 113, x + 166, y + 177), radius=4,
                        fill=(242, 239, 216), outline=(207, 204, 186), width=2)
    d.text((x + 100, y + 125), 'NORTH', fill=(54, 75, 66))
    d.text((x + 104, y + 148), '500 mL', fill=(84, 96, 81))
    d.line((x + 97, y + 63, x + 97, y + 107), fill=(108, 166, 168), width=5)
    if defect == 'scratch':
        d.line((x + 91, y + 86, x + 129, y + 120, x + 120, y + 139),
               fill=(222, 224, 214), width=5)
        d.line((x + 93, y + 87, x + 131, y + 121), fill=(42, 91, 93), width=1)
    elif defect == 'stain':
        d.ellipse((x + 121, y + 151, x + 146, y + 167), fill=(127, 69, 54))
    elif defect == 'label_tear':
        d.polygon([(x + 144, y + 113), (x + 167, y + 113),
                   (x + 167, y + 145), (x + 154, y + 130)], fill=(76, 139, 143))
    elif defect == 'cap_chip':
        d.polygon([(x + 99, y + 31), (x + 116, y + 31),
                   (x + 105, y + 42), (x + 99, y + 43)], fill=(backdrop, backdrop, backdrop - 3))
    if defect and defect not in {'scratch', 'stain', 'label_tear', 'cap_chip'}:
        raise ValueError('Unknown synthetic defect type.')
    return image.filter(ImageFilter.GaussianBlur(radius=0.15))


def generate(root: Path):
    if any(root.glob('**/*.png')):
        raise ValueError(f'{root} already contains PNGs; choose an empty output directory.')
    records = []
    groups = [('reference', 24, False), ('validation', 12, False),
              ('test_normal', 18, False), ('test_defect', 18, True)]
    for group, count, is_defect in groups:
        (root / group).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            seed = 2100 + len(records) * 13
            defect = ['scratch', 'stain', 'label_tear', 'cap_chip'][i % 4] if is_defect else None
            name = f'bottle_{i+1:02d}.png'
            path = root / group / name
            bottle(seed, defect).save(path)
            records.append({'path': f'{group}/{name}', 'split': group,
                            'label': int(is_defect), 'synthetic_defect': defect,
                            'provenance': 'fully synthetic rendered bottle'})
    (root / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(records)} fully synthetic bottle images to {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/demo'))
    generate(parser.parse_args().output)
