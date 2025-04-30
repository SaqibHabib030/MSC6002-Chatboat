# Adaptive Chatbot Training using LoRA and bfloat16 Techniques

This repository contains a Jupyter Notebook used for training a lightweight, adaptive chatbot model using **LoRA (Low-Rank Adaptation)** and **bfloat16** precision techniques. The project explores optimizing transformer-based models with reduced computational cost while retaining performance, suitable for AI-driven educational chatbots.

---

## Project Objective

To fine-tune a large language model using:
- **Parameter-efficient training (LoRA)**
- **Reduced-precision format (bfloat16)**

The goal is to make chatbot training faster, cheaper, and more efficient, especially on limited-resource environments (e.g., consumer GPUs).

---

## Files Included

- `Training.ipynb`: Main notebook for model setup, training, and evaluation.
---

## Key Techniques

- **LoRA**: Reduces number of trainable parameters by injecting low-rank matrices into transformer attention layers.
- **bfloat16**: Optimized floating-point format used to reduce memory usage and accelerate computation without significantly affecting accuracy.

---

## Libraries Used

- `transformers`
- `datasets`
- `peft` (for LoRA)
- `accelerate`
- `torch`
- `scikit-learn`
- `evaluate`
- `wandb` (optional logging)
- `bitsandbytes` (optional for 8-bit optimization)

---

## Usage

1. Clone the repository and open `Training.ipynb`.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook step by step. Ensure your environment supports bfloat16 and/or LoRA (e.g., Colab Pro, A100 GPU).

---

## 📊 Evaluation Metrics

The model is evaluated using:
- **Perplexity**
- **BLEU / ROUGE Scores**
- (Add others used)

---

## Use Case

This notebook was part of a research project in AI-driven education for building a chatbot that adapts learning content based on user interactions. It supports real-time feedback and evolution using fine-tuned transformer models.

---

## Author

**Saqib Habib**  
M.IT in Software Development,  
University of Southern Queensland  

Supervisor: **Dr. Aqeel Sahi**

---
