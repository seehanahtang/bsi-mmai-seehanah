# HW2: Alignment Before Fusion & Multimodal Modeling

**Course:** Multimodal AI (MAS.S60 / 6.S985), MIT Spring 2026
**Notebook:** [`mmai_HW2.ipynb`](../mmai_HW2.ipynb)

---

## Assignment Overview

This homework has two parts:
1. **Reading assignment** on alignment-before-fusion and the Platonic Representation Hypothesis
2. **Modeling homework** applying multimodal fusion strategies

---

## Part 1: Reading Reflection

### Papers

- [Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651) (Li et al., 2021)
- [The Platonic Representation Hypothesis](https://arxiv.org/pdf/2405.07987) (Huh et al., 2024)

### Implications for BSI Prediction

**Align before fuse** suggests that unimodal representations should be brought into a shared semantic space *before* combining them. For BSI prediction, this is highly relevant: tabular EHR features (numeric, binary flags) and clinical note embeddings live in very different spaces. Simple concatenation (early fusion) may not fully capture cross-modal interactions. An alignment step — e.g., projecting both modalities into a shared embedding space, or using cross-attention — could improve fusion quality.

**The Platonic Representation Hypothesis** posits that as models scale, they converge to similar internal representations of reality. This suggests that large pretrained models (ClinicalBERT, Longformer) may already encode representations that are semantically compatible with the structure of tabular clinical data, potentially making alignment less critical at large scale.

---

## Part 2: Modeling

Applied multimodal fusion strategies to the BSI project dataset, exploring early fusion (concatenation) of tabular features and clinical note embeddings, and comparing against unimodal baselines.
