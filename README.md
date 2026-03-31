# Multimodal AI (MAS.S60 / 6.S985) — Course Repository

**Student:** Seehanah Tang (`seehanah@mit.edu`)
**Course:** Multimodal AI, MIT Spring 2026
**Project collaborator:** Carol Gao (`carolgao@mit.edu`)
**Affiliation:** Operations Research Center, MIT

This repository is a living portfolio documenting my work in the MIT Multimodal AI course — assignments, experiments, and the course project on early bloodstream infection detection.

---

## Course Project: Multimodal Early Bloodstream Infection Detection

> **Can multimodal AI — combining structured EHR data with clinical notes — improve early prediction of bloodstream infections before culture results return?**

In collaboration with Hartford Healthcare (the largest hospital system in Connecticut), I am building a multimodal AI framework that fuses tabular EHR features (vitals, ICD-10 diagnoses, medications, chief complaints) with unstructured clinical notes to predict positive blood cultures 1–3 days before results are available. The project explores context-specific modeling across care settings and cross-modal fusion strategies.

See the **[project/README.md](./project/README.md)** for full methodology, architecture diagrams, and results.

---

## Assignments

| # | Topic | Notebook | Notes |
|---|-------|----------|-------|
| [HW1](./hw1/) | Multimodal Data Preprocessing & Embedding Extraction | [`mmai_HW1.ipynb`](./mmai_HW1.ipynb) | Extracted tabular + note embeddings from BSI dataset using Bio_ClinicalBERT |
| [HW2](./hw2/) | Alignment Before Fusion & Multimodal Modeling | [`mmai_HW2.ipynb`](./mmai_HW2.ipynb) | Explored align-before-fuse strategies and Platonic Representation Hypothesis |
| [HW3](./hw3/) | Vision-Language Models & Fine-tuning | [`mmai_HW3.ipynb`](./mmai_HW3.ipynb) | Fine-tuned a VLM; reading on Frozen Language Models and CLIP robustness |

---

## Repository Structure

```
.
├── README.md                        ← You are here
│
├── project/
│   └── README.md                    ← Full project documentation & diagrams
│
├── hw1/README.md                    ← HW1: preprocessing & modality extraction
├── hw2/README.md                    ← HW2: alignment, fusion, and modeling
├── hw3/README.md                    ← HW3: vision-language models
│
├── mmai_HW1.ipynb                   ← HW1 notebook
├── mmai_HW2.ipynb                   ← HW2 notebook
├── mmai_HW3.ipynb                   ← HW3 notebook
│
│── Project notebooks ──────────────────────────────────────────────────────────
├── data_cleaning_labeling.ipynb     ← SQL extraction, cohort building, labeling
├── extract_features.ipynb           ← Vitals, ICD, medications, chief complaints
├── finetune_bsi.ipynb               ← Clinical Longformer fine-tuning on BSI
├── fusion.ipynb                     ← Note truncation + embedding fusion pipeline
├── postprocess.ipynb                ← Embedding postprocessing & dataset merging
│
│── Project scripts ─────────────────────────────────────────────────────────────
├── generate_embeddings.py           ← Bio_ClinicalBERT chunked embedding generation
├── notes_truncation.py              ← Risk-factor-aware clinical note filtering
├── model_battery.py                 ← Multi-model experiment runner (LR/MLP/XGB)
├── model_battery_bsi.py             ← BSI model battery (XGB, TabNet, stacking)
└── models.py                        ← Model definitions and hyperparameter search
```

---

## Quick Links

- [Project README with architecture diagrams](./project/README.md)
- [Data pipeline](./data_cleaning_labeling.ipynb)
- [Feature engineering](./extract_features.ipynb)
- [Text embedding pipeline](./generate_embeddings.py)
- [Multimodal fusion](./fusion.ipynb)
- [Model experiments](./model_battery.py)
