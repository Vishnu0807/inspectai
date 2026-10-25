"""Run one honest hold-out evaluation and an image-condition stress test."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .core import Config, Extractor, Inspector, build_bank, calibrate, heatmap, open_image


def load_split(root, split):
    paths = sorted((root / split).glob('*.png'))
    return [(path.name, open_image(path.read_bytes())) for path in paths]


def metrics(labels, scores, threshold):
    predicted = [int(score > threshold) for score in scores]
    normal_count = sum(label == 0 for label in labels)
    false_alarms = sum(label == 0 and result == 1 for label, result in zip(labels, predicted))
    return {
        'image_auroc': round(float(roc_auc_score(labels, scores)), 4) if len(set(labels)) == 2 else None,
        'precision': round(float(precision_score(labels, predicted, zero_division=0)), 4),
        'recall': round(float(recall_score(labels, predicted, zero_division=0)), 4),
        'false_alarm_rate': round(false_alarms / normal_count, 4) if normal_count else None,
        'false_alarms': false_alarms,
        'normal_count': normal_count,
    }


def altered(image, kind):
    if kind == 'lighting':
        return ImageEnhance.Brightness(image).enhance(0.72)
    if kind == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=1.7))
    if kind == 'viewpoint':
        return image.rotate(8, resample=Image.Resampling.BICUBIC, fillcolor=(235, 235, 232))
    raise ValueError(kind)


def evaluate(root: Path, output: Path):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    assert {row['split'] for row in manifest} == {'reference', 'validation', 'test_normal', 'test_defect'}
    assert len({row['path'] for row in manifest}) == len(manifest)
    refs = load_split(root, 'reference')
    val = load_split(root, 'validation')
    normal = load_split(root, 'test_normal')
    defects = load_split(root, 'test_defect')
    if not all((refs, val, normal, defects)):
        raise ValueError('Every split needs images.')
    config = Config()
    extractor = Extractor(config)
    inspector = Inspector(extractor, build_bank(extractor, [image for _, image in refs], config), config)
    threshold, validation_scores = calibrate(inspector, [image for _, image in val])
    test = [(f'normal/{name}', 0, image) for name, image in normal]
    test += [(f'defect/{name}', 1, image) for name, image in defects]
    records = []
    for name, label, image in test:
        result = inspector.inspect(image)
        records.append({'image': name, 'label': label, 'score': result.score,
                        'decision': 'defect' if result.score > threshold else 'normal',
                        'inference_ms': round(result.seconds * 1000, 2)})
    labels = [record['label'] for record in records]
    scores = [record['score'] for record in records]
    stress = {}
    for kind in ('lighting', 'blur', 'viewpoint'):
        transformed = [inspector.inspect(altered(image, kind)) for _, _, image in test]
        new_scores = [result.score for result in transformed]
        stress[kind] = {
            'change': {'lighting': '72% brightness', 'blur': 'Gaussian radius 1.7 px',
                       'viewpoint': '8° rotation with neutral fill'}[kind],
            'median_normal_score': round(float(median(new_scores[:len(normal)])), 5),
            'median_defect_score': round(float(median(new_scores[len(normal):])), 5),
            'false_alarm_rate': metrics(labels, new_scores, threshold)['false_alarm_rate'],
            'recall': metrics(labels, new_scores, threshold)['recall'],
            'median_inference_ms': round(median(result.seconds * 1000 for result in transformed), 2),
        }
    report = {
        'dataset': 'fully synthetic bottles; synthetic marks, scratches, label tears and cap chips',
        'warning': 'Demonstration only. Scores cannot establish performance on real manufacturing defects.',
        'model': 'torchvision ResNet18_Weights.IMAGENET1K_V1, frozen layer2+layer3',
        'bank_sampling': 'seeded uniform random subset; not the original PatchCore coreset',
        'config': config.__dict__,
        'split_sizes': {name: len(items) for name, items in [('reference', refs), ('validation', val),
                         ('test_normal', normal), ('test_defect', defects)]},
        'threshold': round(threshold, 5),
        'threshold_policy': '95th percentile of separate normal validation image scores; score > threshold is defect',
        'validation_normal_false_alarm_rate': round(float(np.mean(np.asarray(validation_scores) > threshold)), 4),
        'test_metrics': metrics(labels, scores, threshold),
        'baseline_scores': {
            'median_normal_score': round(float(median(scores[:len(normal)])), 5),
            'median_defect_score': round(float(median(scores[len(normal):])), 5),
        },
        'median_inference_ms': round(float(median(record['inference_ms'] for record in records)), 2),
        'stress_test': stress,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    example = inspector.inspect(defects[0][1])
    heatmap(defects[0][1], example.patch_map, threshold).save(output.with_name('example_heatmap.png'))
    print(json.dumps({key: report[key] for key in ('split_sizes', 'threshold', 'test_metrics', 'median_inference_ms', 'stress_test')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/demo'))
    parser.add_argument('--output', type=Path, default=Path('reports/demo_metrics.json'))
    args = parser.parse_args()
    evaluate(args.data, args.output)
