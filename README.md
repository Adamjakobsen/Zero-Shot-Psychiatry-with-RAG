# Knowledge-Guided RAG for Zero-Shot Psychiatric Data: Privacy Preserving Synthetic Data Generation

Access to real patient data is a significant bottleneck in clinical AI research. This project presents a zero-shot, knowledge-guided framework for generating high-fidelity, privacy-preserving synthetic psychiatric data.

Instead of training on sensitive patient records, our method steers a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG). The knowledge base for retrieval is built from established clinical manuals: the Diagnostic and Statistical Manual of Mental Disorders (DSM-5) and the International Classification of Diseases (ICD-10). This "privacy-by-design" approach ensures that the generative process is entirely decoupled from real data, eliminating the risk of memorization or leakage.

We benchmark our LLM-based models against traditional data-trained generative models (CTGAN, TVAE) and evaluate them across six anxiety-related disorders. The evaluation focuses on two key dimensions:

Fidelity: How well the synthetic data captures the statistical properties of real data.

Privacy: The risk of re-identifying individuals from the synthetic dataset.

The results demonstrate that knowledge-augmented generation is a viable alternative for creating realistic clinical data, especially in scenarios where data access is highly restricted.

## Requirements

Python 3.9 or higher.

Ollama installed and running for local LLM inference. Ensure you have pulled a model (e.g., ollama pull mistral).

```
pandas
numpy
scikit-learn
sdv
pyyaml
tqdm
requests
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
```

## Datasets
All the datasets can be found under `datasets/`.

## 5. How to Run the Experiments

The experimental pipeline is divided into several stages. Run the scripts in the following order.

### Step 1: Generate LLM-Based Synthetic Data
This script uses the RAG framework to generate synthetic data for the LLM variants (`dsm5`, `icd10`, `dsm5+icd10`, `none`).

```bash
python main.py
```

### Step 2: Generate Baseline Synthetic Data
This script generates data for the baseline models (`ctgan`, `tvae`, `random`).

```bash
python generate_baselines.py
```

After this step, the `datasets/synthetic/` directory will be populated with all required data files.

### Step 3: Run the Fidelity Evaluation
This script calculates the fidelity metrics (`JSD`, `MAE_V`, `ED²`) for all generated datasets and produces bootstrapped confidence intervals.

```bash
python evaluate_fidelity.py
```

This will create `fidelity_summary.csv` and `fidelity_raw_bootstrap.csv` in the `results/fidelity/` directory.

### Step 4: Run the Privacy Evaluation
This script calculates the privacy metrics (`ExactOverlap`, `dNN`, `Share`, `kmap`) for all generated datasets.

```bash
python evaluate_privacy.py
```

This will create `privacy_summary.csv` and the final LaTeX table `privacy_results.tex` in the `results/privacy/` directory.

### Step 5: Generate Final LaTeX Reports (Fidelity)
These scripts use the outputs from the fidelity evaluation to generate the final, formatted LaTeX tables for the paper.

```bash
# Generates the main fidelity results table
python fidelity_report.py

# Generates the ablation study (delta) table
python delta_CI.py
```

These scripts will create `fidelity_results.tex` and `fidelity_delta_results.tex` in the `results/fidelity/` directory.
