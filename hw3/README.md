# HW3: Vision-Language Models & Fine-tuning

**Course:** Multimodal AI (MAS.S60 / 6.S985), MIT Spring 2026
**Notebook:** [`mmai_HW3.ipynb`](../mmai_HW3.ipynb)

---

## Assignment Overview

This homework explores Vision-Language Models (VLMs) with hands-on fine-tuning experience on GPU (A100 via Colab Pro).

---

## Part 1: Reading Reflection

### Papers

- [Multimodal Few-Shot Learning with Frozen Language Models](https://arxiv.org/pdf/2106.13884) (Tsimpoukelli et al., 2021)
- [Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP](https://arxiv.org/pdf/2208.05516.pdf)
- [Generative AI: Here to stay, but for good?](https://www.sciencedirect.com/science/article/pii/S0160791X2300177X)

### Key Takeaways

**Frozen LMs for few-shot multimodal learning:** By keeping the language model frozen and only training a vision encoder, Tsimpoukelli et al. show that VLMs can adapt to new vision-language tasks with very few examples. This has direct relevance for clinical settings where labeled data is scarce — a frozen clinical LM with a fine-tuned vision encoder could generalize to new diagnostic tasks.

**CLIP robustness:** Dataset curation quality matters more than raw scale. Noisy web-scraped data can degrade robustness, especially for distribution shifts. In healthcare, this maps to the importance of curating high-quality clinical image-text pairs rather than relying on large noisy EHR corpora.

---

## Part 2: VLM Fine-tuning

Hands-on fine-tuning of a vision-language model, with GPU verification, model loading, and evaluation on a downstream task.

**Environment:** Google Colab A100 GPU
**Libraries:** `transformers`, `accelerate`, `bitsandbytes`, `torch`
