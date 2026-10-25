# InspectAI

InspectAI compares a bottle image with *normal* reference examples. A frozen pretrained ResNet-18 produces a small grid of local features. Each test patch is compared with the closest patch in a memory of normal examples. The largest distance becomes the image score; the patch distances form a heatmap. The decision threshold comes from a **different set of normal validation images**.

This is a compact **PatchCore-inspired** application, not a reproduction of the full paper: it uses a seeded random subset of normal features instead of greedy coreset selection, and it omits PatchCore's score reweighting. The network is pretrained on ImageNet; no neural network is trained here.

## Run the website (Windows PowerShell)

Install Python 3.12, unzip the project, and open PowerShell in the folder containing `app.py`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL shown by Streamlit, usually <http://localhost:8501>. The **first run downloads about 45 MB of official ResNet-18 weights** and builds the bank, which can take a minute. Later runs reuse the downloaded weights and Streamlit's in-process bank cache. A CPU is enough; no GPU or API key is required. Press Ctrl+C to stop the server.

On Windows, if `py -3.12` is not available, install Python 3.12 and reopen PowerShell. The requirements request matching **CPU** PyTorch and torchvision builds from PyTorch's package index. If you already have a working local CUDA installation, use an isolated environment before changing those pins.

The bundled example works immediately after setup. It includes **24 normal references, 12 different normal validation images, 18 normal test images, and 18 synthetically defective test images**. Every image was drawn by `generate_demo.py`, and its label/provenance is recorded in `data/demo/manifest.json`. There is no borrowed industrial dataset and no claim about real factory performance.

## Try it

1. Choose a bundled test image, such as `test_defect/bottle_01.png`, and click **Run inspection**. Compare the original with the anomaly overlay.
2. Choose several files under **Upload image or batch** and download scores, thresholds, decisions, and processing times as CSV.
3. Expand **Reviewer feedback**, record the actual normal/defect label and an optional note, and export reviews as CSV.
4. Open **Evaluation & robustness** to see the held-out metrics and what happened under lighting, blur, and viewpoint changes.

To inspect your own product, select **Upload your own** in the sidebar. Supply 3–100 *normal* reference photos and at least 5 **different normal** validation photos, preferably from a consistent camera setup. Then upload one or more inspection images. Do not upload test or defective images into the reference or normal validation sets. The app rejects byte-identical images appearing in both sets. It cannot detect near-duplicate photos or mislabeled images; you need to keep those splits clean.

The reviewer feedback database stores a SHA-256 image hash, source name, score, prediction, actual label, and note. **It does not store the uploaded picture or automatically update the model.** The SQLite file is local (`data/feedback.sqlite3`) and excluded from Git. On hosting with ephemeral storage, export the reviews before a restart; durable multi-user review needs an external database.

## Evaluation

The checked-in report was produced with real pretrained ResNet-18 weights:

| Held-out synthetic bottles | Value |
| --- | ---: |
| Image AUROC | 1.000 |
| Precision | 0.818 |
| Recall | 1.000 |
| False-alarm rate (normal test images) | 0.222 (4/18) |
| Median model + nearest-neighbour time | About 65 ms on the evaluation machine |

The threshold is the **95th percentile of the 12 normal validation scores**. AUROC uses test scores; precision, recall, and false-alarm rate use that fixed threshold. A score *greater* than the threshold is predicted defective. Time excludes the first weight download and bank construction. Timing varies by machine.

The robustness comparison applies changes to the **held-out test images after fixing the threshold**:

| Condition | Normal false-alarm rate | Defect recall |
| --- | ---: | ---: |
| Original images | 22.2% | 100% |
| 72% brightness | 22.2% | 100% |
| Gaussian blur, radius 1.7 px | 100% | 100% |
| 8° rotation | 100% | 100% |

The high AUROC comes from an easy, fully synthetic example. Blur and rotation wrongly flag every changed normal bottle, a useful demonstration that *ranking defects above normals* does not imply a stable fixed threshold under image shifts. This is a diagnostic demo, not a validated manufacturing system. The heatmap uses a 14×14 grid enlarged over the image; it is a visual cue, not a pixel-accurate defect segmentation mask.

Recreate the example data and evaluation from an **empty** output directory:

```powershell
.\.venv\Scripts\python.exe generate_demo.py --output data\my_demo
.\.venv\Scripts\python.exe -m inspectai.evaluate --data data\my_demo --output reports\my_demo_metrics.json
```

Or rerun the checked-in sample (which updates its report):

```powershell
.\.venv\Scripts\python.exe -m inspectai.evaluate --data data\demo --output reports\demo_metrics.json
```

That command writes JSON metrics, per-image CSV scores, and an example heatmap. To evaluate **real** data, prepare distinct normal reference images, normal validation images, and labelled normal/defect test images in the same folder structure; update the manifest provenance honestly. Report performance separately from synthetic results. Never choose the threshold using the test labels.

## Design and files

```text
app.py                    Streamlit inspect, batch export, reviewer, evaluation UI
inspectai/core.py         Frozen CNN, feature memory, nearest neighbours, heatmap
inspectai/evaluate.py     Held-out image metrics and robustness stress test
inspectai/feedback.py     Local SQLite reviewer notes (no raw image storage)
generate_demo.py          Reproducible fully synthetic bottle renders
data/demo/                Separate reference / validation / test folders + manifest
reports/demo_metrics.json Measured example results; CSV and heatmap accompany it
tests/                    Logic, input and data-provenance tests
```

**Feature path:** ImageNet normalization and 224×224 input → ResNet-18 `layer2` and `layer3` → pooled 14×14 grid → per-layer vector normalization → concatenation → at most 768 reference vectors sampled with a fixed seed → Euclidean 1-nearest-neighbour distance for every patch → maximum distance for the image score. The bank and threshold are cached per reference set in the app process. A different camera angle, product type, or lighting may require a new bank and validation set.

