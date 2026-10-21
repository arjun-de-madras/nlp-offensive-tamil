# Offensive Language Detection in Tamil–English Code-Mixed Text

UCS3564 Natural Language Processing — Micro Project (Sem V, 2026–27 odd)
SSN College of Engineering, Department of CSE

An end-to-end system that takes a raw social-media comment (romanized Tamil / "Tanglish", Tamil script or mixed)
and classifies it into six classes: *not offensive, offensive untargeted, offensive targeted at an individual /
a group / another target,* and *not Tamil*. It combines input validation and normalisation, a fine-tuned
**MuRIL** transformer (class-weighted loss), a TF-IDF + LinearSVC baseline, sliding-window aggregation for long
inputs, confidence-based abstention and a script guard. Served as a Gradio web app with a REST API.

Full write-up: [`report.pdf`](report.pdf).

## Results (held-out test set, 4,388 comments, seed 42)

| Metric | TF-IDF + LinearSVC | MuRIL (fine-tuned) |
|---|---|---|
| Accuracy | 0.767 | 0.748 |
| Macro-F1 | **0.499** | 0.397 |
| Weighted-F1 | **0.767** | 0.728 |

McNemar test (transformer vs baseline): p = 1.7 × 10⁻³. On this 3-epoch, single-T4 run the transformer did **not**
beat the baseline (IND and OTH collapse to zero recall); validation macro-F1 was still rising when early stopping
ended training. See the report (Sections VII, XIII) for the analysis. Latency: p95 12.5 ms (T4 GPU), 130 ms (CPU, 1 thread).

## Quick start

```bash
git clone <your-repo-url> && cd nlp-offensive-tamil
bash run.sh            # installs deps, runs the whole pipeline with seed 42, launches the demo
bash run.sh demo       # demo only (needs ./artifacts from a previous run)
bash run.sh --share    # full run + public gradio.live link
```

**Colab (recommended):** open `NLP_MicroProject_Offensive_Tamil.ipynb`, set *Runtime → T4 GPU*, *Run all*.
Set `fast_mode=True` in the config cell for a ~6 min smoke test (lower scores).

**REST API** (while the app is running):
```python
from gradio_client import Client
Client("http://127.0.0.1:7860").predict("text here", "MuRIL transformer (fine-tuned)", api_name="/classify")
```

## Reproducibility

* Fixed random seed **42** (`CFG["seed"]`): `set_seed()` seeds `random`, `numpy`, `torch`, CUDA, sets
  `cudnn.deterministic=True` and `PYTHONHASHSEED`; `run.sh` also exports `PYTHONHASHSEED=42`.
* The train/val/test split is stratified with seed 42 and stored as CSVs in `data/`.
* Exact results can still differ slightly across GPU types / library versions (fp16 mixed precision).
* The ~1 GB fine-tuned weights are not in git (`artifacts/transformer/`); `run.sh` regenerates them.

## Dataset

Offenseval Dravidian (Tamil), Chakravarthi et al., DravidianLangTech @ EACL 2021, loaded from Hugging Face
`community-datasets/offenseval_dravidian` (config `tamil`). **Licence: CC BY 4.0.**

## Repository structure

```
NLP_MicroProject_Offensive_Tamil.ipynb   full pipeline: problem → data → baseline → transformer → evaluation → edge cases → tests → demo
nlp_core.py        preprocessing, script ID, input guard, predictors (sliding window + abstention)
app.py             Gradio UI + REST API
run.sh             reproducible run script
requirements.txt   dependencies
report.pdf         IEEE-format project report
data/              fixed train / val / test split (CSV)
artifacts/         baseline.joblib, inference_config.json, MODEL_CARD.md, error/edge-case/test CSVs
figures/           architecture, data overview, training curves, confusion matrices, selective prediction
results.json       all metrics, latency, memory, tests, success criteria
```

## Declaration of AI-assistant use

As required by the course: parts of the code and text of this project were drafted with an AI assistant
(Anthropic Claude). All code was run, checked and edited by the team, and every member can explain the complete system.

## Team

| Member | Register no. | Contribution |
|---|---|---|
| Arjun Balaji | 3122245001022 | _TODO_ |
| Harsha Harirajan | 3122245001056 | _TODO_ |

## References

1. Chakravarthi et al. Findings of the Shared Task on Offensive Language Identification in Tamil, Malayalam, and Kannada. DravidianLangTech @ EACL 2021.
2. Chakravarthi et al. DravidianCodeMix. Language Resources and Evaluation, 2022.
3. Jayanthi & Gupta. SJ_AJ@DravidianLangTech-EACL2021. 2021.
4. Hande et al. Offensive Language Identification in Low-resourced Code-mixed Dravidian Languages using Pseudo-labeling. arXiv, 2021.
5. Khanuja et al. MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730, 2021.
6. Kakwani et al. IndicNLPSuite. Findings of EMNLP 2020.
7. Conneau et al. Unsupervised Cross-lingual Representation Learning at Scale. ACL 2020.
8. Ribeiro et al. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL 2020.
