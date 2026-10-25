"""Streamlit interface for one-category, PatchCore-inspired bottle inspection."""
import csv
from hashlib import sha256
from io import StringIO
from pathlib import Path

import streamlit as st

from inspectai.core import Config, Extractor, Inspector, MODEL_NAME, build_bank, calibrate, heatmap, open_image
from inspectai.feedback import export_csv, record


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'demo'
FEEDBACK = ROOT / 'data' / 'feedback.sqlite3'
CONFIG = Config()
st.set_page_config(page_title='InspectAI · Bottle inspection', page_icon='◉',
                   layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
    .block-container{max-width:1120px;padding-top:2.6rem}
    [data-testid="stMetric"]{background:#f8f9f6;border:1px solid #dfe5dd;padding:15px;border-radius:5px}
    [data-testid="stImage"] img{border:1px solid #e3e7e0;border-radius:5px}
    .tag{font:600 11px ui-monospace;letter-spacing:.13em;color:#67816b;text-transform:uppercase}
    .lede{color:#68746c;max-width:690px;line-height:1.6}
    .small-note{font-size:12px;color:#68746c}
    </style><div class="tag">VISUAL QUALITY CONTROL / BOTTLES</div>''', unsafe_allow_html=True)
st.title('InspectAI')
st.markdown('<p class="lede">Compare a bottle image with examples of normal products. See the unusual patches, then check the result against a threshold set with separate validation images.</p>', unsafe_allow_html=True)


def sample_bytes(split):
    return tuple(path.read_bytes() for path in sorted((DATA / split).glob('*.png')))


@st.cache_resource(show_spinner='Loading pretrained ResNet-18 weights…')
def load_extractor():
    return Extractor(CONFIG)


@st.cache_resource(show_spinner='Building reference bank and calibrating threshold…')
def prepare(reference: tuple[bytes, ...], validation: tuple[bytes, ...]):
    extractor = load_extractor()
    refs = [open_image(data) for data in reference]
    vals = [open_image(data) for data in validation]
    inspector = Inspector(extractor, build_bank(extractor, refs, CONFIG), CONFIG)
    threshold, scores = calibrate(inspector, vals)
    return inspector, threshold, scores


with st.sidebar:
    st.subheader('Reference set')
    source = st.radio('Normal examples', ['Bundled synthetic bottles', 'Upload your own'],
                      help='Use images from one product category and consistent camera setup.')
    if source == 'Upload your own':
        refs_upload = st.file_uploader('Normal reference images (3–100)', type=['png', 'jpg', 'jpeg'],
                                       accept_multiple_files=True, key='refs')
        vals_upload = st.file_uploader('Different normal validation images (at least 5)',
                                       type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key='vals')
        reference = tuple(file.getvalue() for file in refs_upload or ())
        validation = tuple(file.getvalue() for file in vals_upload or ())
    else:
        reference = sample_bytes('reference')
        validation = sample_bytes('validation')
    st.divider()
    st.caption(f'Model: {MODEL_NAME}')
    st.caption(f'Grid: {CONFIG.grid} × {CONFIG.grid} · bank cap: {CONFIG.bank_size} patches')
    st.caption('Model downloads once on first use. CPU inference; no model fine-tuning.')

if len(reference) < 3 or len(validation) < 5:
    st.info('Add at least 3 normal reference images and 5 *different* normal validation images to begin.')
    st.stop()
if len(reference) > 100 or len(validation) > 100:
    st.error('Use at most 100 reference images and 100 validation images.')
    st.stop()
if {sha256(data).hexdigest() for data in reference} & {sha256(data).hexdigest() for data in validation}:
    st.error('The same image appears in reference and validation. Keep these sets separate.')
    st.stop()
if sum(map(len, reference + validation)) > 60 * 1024 * 1024:
    st.error('Combined reference and validation uploads exceed 60 MB.')
    st.stop()

try:
    inspector, threshold, val_scores = prepare(reference, validation)
except Exception as exc:
    st.error(f'Could not prepare the model: {exc}')
    st.info('Check the image formats and network access for the first pretrained-weight download.')
    st.stop()
reference_fingerprint = sha256(b''.join(sha256(data).digest() for data in reference + validation)).hexdigest()

cols = st.columns(3)
cols[0].metric('Normal reference images', len(reference))
cols[1].metric('Validation normals', len(validation))
cols[2].metric('Threshold', f'{threshold:.3f}')

inspect_tab, evaluation_tab = st.tabs(['Inspect', 'Evaluation & robustness'])
with inspect_tab:
    if source == 'Bundled synthetic bottles':
        st.caption('Sample images are generated, fully synthetic bottle renders. Defects are synthetic alterations.')
    option = st.radio('Choose images', ['Bundled test image', 'Upload image or batch'], horizontal=True)
    if option == 'Bundled test image':
        choices = sorted((DATA / 'test_normal').glob('*.png')) + sorted((DATA / 'test_defect').glob('*.png'))
        chosen = st.selectbox('Example', choices, format_func=lambda p: f'{p.parent.name} / {p.name}')
        candidates = [(f'{chosen.parent.name}/{chosen.name}', chosen.read_bytes())] if choices else []
    else:
        uploads = st.file_uploader('PNG or JPEG images (up to 30)', type=['png', 'jpg', 'jpeg'],
                                   accept_multiple_files=True, key='inspections')
        candidates = [(file.name, file.getvalue()) for file in uploads or ()]
    if st.button('Run inspection', type='primary'):
        if not candidates:
            st.warning('Choose at least one image.')
        elif len(candidates) > 30:
            st.error('Use at most 30 inspection images in one batch.')
        else:
            results = []
            for name, data in candidates:
                try:
                    image = open_image(data)
                    result = inspector.inspect(image)
                    results.append({'name': name, 'hash': sha256(data).hexdigest(),
                                    'data': data, 'score': result.score,
                                    'map': result.patch_map, 'ms': result.seconds * 1000,
                                    'prediction': 'defect' if result.score > threshold else 'normal'})
                except (ValueError, OSError) as exc:
                    st.warning(f'{name}: {exc}')
            st.session_state['inspection_results'] = results
            st.session_state['inspection_reference'] = reference_fingerprint
    results = st.session_state.get('inspection_results', [])
    if results and st.session_state.get('inspection_reference') != reference_fingerprint:
        st.warning('The reference set changed. Run inspection again to update these results.')
        results = []
    if results:
        st.caption(f'{len(results)} inspected · Threshold {threshold:.3f} · Lower scores resemble the normal references.')
        if len(results) > 1:
            selected = st.selectbox('View result', range(len(results)),
                                    format_func=lambda i: f'{results[i]["name"]} · {results[i]["prediction"]}')
        else:
            selected = 0
        item = results[selected]
        st.subheader(item['name'])
        status = 'Review as unusual' if item['score'] > threshold else 'Within normal range'
        st.write(f'**{status}** · score `{item["score"]:.3f}` · {item["ms"]:.1f} ms model + nearest-neighbour time')
        left, right = st.columns(2)
        left.image(open_image(item['data']), caption='Original image', width='stretch')
        right.image(heatmap(open_image(item['data']), item['map'], threshold),
                    caption='Patch distance overlay · brighter yellow = farther from normal', width='stretch')
        st.caption('The heatmap shows relative feature distance, not a pixel-perfect defect boundary or a calibrated probability.')
        out = StringIO()
        writer = csv.writer(out)
        writer.writerow(['image', 'sha256', 'score', 'threshold', 'decision', 'inference_ms'])
        for row in results:
            writer.writerow([row['name'], row['hash'], f'{row["score"]:.6f}',
                             f'{threshold:.6f}', row['prediction'], f'{row["ms"]:.2f}'])
        st.download_button('Download results (CSV)', out.getvalue(), 'inspectai_results.csv', 'text/csv')
        with st.expander('Reviewer feedback'):
            label = st.radio('Actual inspection label', ['normal', 'defect'], horizontal=True)
            note = st.text_input('Short note (optional)', max_chars=300, placeholder='e.g. glare, harmless scratch, missed stain')
            if st.button('Record review'):
                try:
                    record(FEEDBACK, item['hash'], item['name'][:100], item['score'], threshold,
                           item['prediction'], label, note)
                    st.success('Review recorded. Feedback does not automatically change the bank or threshold.')
                except (OSError, ValueError) as exc:
                    st.error(f'Could not save review: {exc}')
            st.download_button('Export reviews (CSV)', export_csv(FEEDBACK), 'inspectai_reviews.csv', 'text/csv')

with evaluation_tab:
    st.subheader('Hold-out evaluation')
    st.caption('These metrics apply only to the bundled synthetic reference set, regardless of the current sidebar selection. Reference, validation, and test renders are separate.')
    report_path = ROOT / 'reports' / 'demo_metrics.json'
    if report_path.exists():
        import json
        report = json.loads(report_path.read_text(encoding='utf-8'))
        metrics = report['test_metrics']
        a, b, c, d = st.columns(4)
        a.metric('Image AUROC', f'{metrics["image_auroc"]:.3f}')
        b.metric('Precision', f'{metrics["precision"]:.3f}')
        c.metric('Recall', f'{metrics["recall"]:.3f}')
        d.metric('False-alarm rate', f'{metrics["false_alarm_rate"]:.1%}')
        st.caption(f'Median inference: {report["median_inference_ms"]:.1f} ms · test: '
                   f'{report["split_sizes"]["test_normal"]} normal and '
                   f'{report["split_sizes"]["test_defect"]} altered images.')
        st.subheader('What changes under image conditions?')
        table = [{'Condition': 'Original', 'Change applied': 'None',
                  'Median normal score': report['baseline_scores']['median_normal_score'],
                  'False-alarm rate': metrics['false_alarm_rate'], 'Defect recall': metrics['recall']}]
        table += [{'Condition': name, 'Change applied': row['change'],
                  'Median normal score': row['median_normal_score'],
                  'False-alarm rate': row['false_alarm_rate'], 'Defect recall': row['recall']}
                 for name, row in report['stress_test'].items()]
        st.dataframe(table, width='stretch', hide_index=True)
        st.caption('Stress changes are applied only after the threshold is fixed. Higher normal scores can mean false alarms.')
    else:
        st.info('Evaluation report not generated yet. Run `python -m inspectai.evaluate` from the project folder.')
    st.markdown('**Method:** Frozen ImageNet ResNet-18 → local layer2/layer3 features → seeded random normal-patch bank → nearest-neighbour distance. The maximum patch distance is the image score. The decision threshold is the 95th percentile of separate normal validation scores.')
    st.caption('This is a simplified PatchCore-style implementation. It does not include PatchCore’s greedy coreset or score reweighting, and the synthetic results do not estimate performance on real factory images.')
