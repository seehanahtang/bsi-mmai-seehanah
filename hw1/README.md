# HW1: Multimodal Data Preprocessing & Embedding Extraction

**Course:** Multimodal AI (MAS.S60 / 6.S985), MIT Spring 2026
**Notebook:** [`mmai_HW1.ipynb`](../mmai_HW1.ipynb)

---

## Assignment Overview

The first homework asks us to choose a dataset and project objective, then demonstrate preprocessing of multiple modalities to prepare for multimodal learning.

---

## My Approach: BSI Dataset

I applied this to my course project dataset — early bloodstream infection (BSI) detection from Hartford Healthcare EHR data.

### Modalities Extracted

| Modality | Description | Preprocessing |
|----------|-------------|---------------|
| **Tabular** | ICD-10 diagnosis codes, chief complaints, medication flags, vital signs | Binary flags, aggregation, imputation |
| **Clinical Notes** | H&P and nurse notes preceding blood culture order | Truncated to BSI-relevant sentences, then embedded with Bio_ClinicalBERT |

### Text Embedding Pipeline

Notes are processed using `Bio_ClinicalBERT` (`emilyalsentzer/Bio_ClinicalBERT`):

```
Raw clinical note
    → Risk-factor-aware truncation (notes_truncation.py)
    → Tokenize with ClinicalBERT tokenizer
    → Split into 512-token chunks (with padding)
    → Embed each chunk independently
    → Mean pool across chunks
    → 768-dimensional note embedding
```

### Dataset Stats After Preprocessing

- ~44,963 encounters (2022), ~45,766 (2023), ~39,603 (2024)
- Positive blood culture rate: ~9% across all years
- ~3,500 total feature dimensions after joining tabular + embeddings

---

## Key Takeaways

- Clinical notes require domain-aware preprocessing: boilerplate removal and relevance filtering significantly reduce noise before embedding
- Chunked mean-pooling is an effective way to handle notes that exceed BERT's 512-token limit
- The 9% positive rate requires careful class balancing during model training
