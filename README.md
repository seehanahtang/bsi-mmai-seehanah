# Multimodal AI (MAS.S60 / 6.S985) — Seehanah Tang

**Student:** Seehanah Tang (`seehanah@mit.edu`)
**Course:** Multimodal AI, MIT Spring 2026
**Project collaborator:** Carol Gao (`carolgao@mit.edu`)
**Affiliation:** Operations Research Center, MIT

## Bio
<img src="./imgs/profile.png" style="width:200px;">

Hey! I am a first-year PhD student at the MIT [Operations Research Center](https://orc.mit.edu) advised by Professor [Dimitris Bertsimas](https://www.dbertsim.mit.edu). My research sits at the intersection of optimization, machine learning, and healthcare, with a focus on building data-driven tools that meaningfully improve patient outcomes and clinical operations. Outside of research, I play flute in the MIT Symphony Orchestra, am an avid traveler, and enjoy strategy card games.

---

## Course Project: Multimodal Early Bloodstream Infection Detection

> **Can multimodal AI — combining structured EHR data with clinical notes — improve early prediction of bloodstream infections before culture results return?**

In collaboration with Hartford Healthcare (the largest hospital system in Connecticut), I am building a multimodal AI framework that fuses tabular EHR features (vitals, ICD-10 diagnoses, medications, chief complaints) with unstructured clinical notes to predict positive blood cultures 1–3 days before results are available. The project explores context-specific modeling across care settings and cross-modal fusion strategies.

See the **[project/README.md](./project/README.md)** for full methodology, architecture diagrams, and results.

---

## Assignments

| # | Topic | Notebook | Notes |
|---|-------|----------|-------|
| [HW1](./hw1/) | Multimodal Data Preprocessing & Embedding Extraction | [`hw1/mmai_HW1.ipynb`](./hw1/mmai_HW1.ipynb) | Extracted tabular + note embeddings from BSI dataset using Bio_ClinicalBERT |
| [HW2](./hw2/) | Alignment Before Fusion & Multimodal Modeling | [`hw2/mmai_HW2.ipynb`](./hw2/mmai_HW2.ipynb) | Explored align-before-fuse strategies and Platonic Representation Hypothesis |
| [HW3](./hw3/) | Vision-Language Models & Fine-tuning | [`hw3/mmai_HW3.ipynb`](./hw3/mmai_HW3.ipynb) | Fine-tuned a VLM; reading on Frozen Language Models and CLIP robustness |

---

## Repository Structure

```
.
├── README.md                        ← You are here
│
├── project/
│   ├── README.md                    ← Full project documentation & diagrams
│   │
│   │── Notebooks ───────────────────────────────────────────────────────────────
│   ├── data_cleaning_labeling.ipynb ← SQL extraction, cohort building, labeling
│   ├── extract_features.ipynb       ← Vitals, ICD, medications, chief complaints
│   ├── finetune_bsi.ipynb           ← Clinical Longformer fine-tuning on BSI
│   ├── fusion.ipynb                 ← Embedding fusion pipeline
│   ├── postprocess.ipynb            ← Embedding postprocessing & dataset merging
│   │
│   │── Scripts ─────────────────────────────────────────────────────────────────
│   ├── generate_embeddings.py       ← Bio_ClinicalBERT chunked embedding generation
│   ├── notes_truncation.py          ← Risk-factor-aware note filtering (experimental; did not improve results)
│   ├── model_battery.py             ← Multi-model experiment runner (LR/MLP/XGB)
│   ├── model_battery_bsi.py         ← BSI model battery (XGB, TabNet, stacking; results so far use XGB only)
│   └── models.py                    ← Model definitions and hyperparameter search
│
├── hw1/
│   ├── README.md                    ← HW1: preprocessing & modality extraction
│   └── mmai_HW1.ipynb
├── hw2/
│   ├── README.md                    ← HW2: alignment, fusion, and modeling
│   └── mmai_HW2.ipynb
└── hw3/
    ├── README.md                    ← HW3: vision-language models
    └── mmai_HW3.ipynb
```

---

## Quick Links

- [Project README with architecture diagrams](./project/README.md)
- [Data pipeline](./project/data_cleaning_labeling.ipynb)
- [Feature engineering](./project/extract_features.ipynb)
- [Text embedding pipeline](./project/generate_embeddings.py)
- [Multimodal fusion](./project/fusion.ipynb)
- [Model experiments](./project/model_battery.py)
