from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from inspectai.core import Config, Inspector, build_bank, calibrate, heatmap, open_image
from inspectai.evaluate import altered, metrics
from inspectai.feedback import export_csv, record


class FakeExtractor:
    """Deterministic patch features test the pipeline without downloading model weights."""

    def patches(self, image):
        colour = np.asarray(image).mean(axis=(0, 1))[:2] / 255
        return np.tile(colour.astype('float32'), (4, 1))


def solid(rgb):
    return Image.new('RGB', (64, 64), rgb)


def test_build_score_calibrate_and_heatmap():
    config = Config(grid=2, bank_size=8)
    normal = solid((120, 120, 120))
    unusual = solid((240, 15, 15))
    bank = build_bank(FakeExtractor(), [normal] * 3, config)
    inspector = Inspector(FakeExtractor(), bank, config)
    threshold, scores = calibrate(inspector, [solid((121 + i, 120, 120)) for i in range(5)])
    assert threshold > 0
    assert len(scores) == 5
    assert inspector.inspect(unusual).score > threshold
    assert inspector.inspect(unusual).patch_map.shape == (2, 2)
    assert heatmap(unusual, inspector.inspect(unusual).patch_map, threshold).size == (320, 320)


def test_multi_image_failures_do_not_yield_partial_bank():
    with pytest.raises(ValueError, match='between 3 and 100'):
        build_bank(FakeExtractor(), [solid((0, 0, 0))])
    with pytest.raises(ValueError, match='five'):
        calibrate(Inspector(FakeExtractor(), np.ones((3, 2), dtype='float32'), Config(grid=2)), [solid((1, 2, 3))])


def test_open_image_and_reject_invalid_data():
    buffer = BytesIO()
    solid((1, 2, 3)).save(buffer, format='PNG')
    assert open_image(buffer.getvalue()).mode == 'RGB'
    with pytest.raises(ValueError, match='Could not read'):
        open_image(b'this is not an image')
    with pytest.raises(ValueError, match='8 MB'):
        open_image(b'a' * (8 * 1024 * 1024 + 1))


def test_metrics_and_robustness_operations():
    result = metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert result['false_alarm_rate'] == 0.5
    assert result['recall'] == 0.5
    assert result['image_auroc'] == 0.75
    for condition in ('lighting', 'blur', 'viewpoint'):
        assert altered(solid((120, 120, 120)), condition).size == (64, 64)


def test_reviewer_feedback_export(tmp_path):
    db = tmp_path / 'reviews.sqlite3'
    record(db, 'a' * 64, 'sample.png', 0.4, 0.3, 'defect', 'normal', 'lighting issue')
    exported = export_csv(db)
    assert 'lighting issue' in exported
    assert 'sample.png' in exported
    assert len(exported.splitlines()) == 2
    with pytest.raises(ValueError, match='Invalid'):
        record(db, 'a' * 64, 'sample.png', 0.4, 0.3, 'defect', 'unknown', '')


def test_demo_splits_are_disjoint_and_synthetic():
    root = Path(__file__).resolve().parents[1] / 'data' / 'demo'
    import json
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    assert len(manifest) == 72
    assert len({row['path'] for row in manifest}) == 72
    assert {row['split'] for row in manifest} == {'reference', 'validation', 'test_normal', 'test_defect'}
    assert all(row['provenance'].startswith('fully synthetic') for row in manifest)
    assert all(row['label'] == 0 for row in manifest if row['split'] in ('reference', 'validation'))
