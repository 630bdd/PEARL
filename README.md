# PEARL: Learning to Paraphrase for Alignment with LLM Preference

## Overview

Large Language Models (LLMs) suffer from **paraphrase divergence** - semantically similar questions phrased differently may elicit incorrect responses despite correct answers to original questions. PEARL addresses this by:

✅ Learning to paraphrase questions into model-preferred expressions  
✅ Black-box approach requiring no model retraining  
✅ Validated across 6 datasets covering QA, commonsense reasoning, and math problems

For more details, please refer to our paper:
[Junbo Fu, Guoshuai Zhao*, Yimin Deng, Yunqi Mi, Xueming Qian](https://aclanthology.org/2024.findings-emnlp.134/)

---

## Installation and Setup

### 1. Environment Setup
Ensure you have **Conda** installed. Then, create a new environment and install dependencies:
```bash
conda create -n pearl_env python=3.10
conda activate pearl_env
pip install -r requirements.txt
```

### 2. Download Required Data and Models
- **Data**: Download PEARL generator training data, test data, and auxiliary data for baselines from: `[Insert Data URL]`
- **Model**: Download the pre-trained model from: `[Insert Model URL]`

---

## Usage

### 1. Train PEARL Generator
To train the PEARL generator, run:
```bash
bash train.sh
```
Alternatively, you can use our pre-trained model available at: `[Insert Pre-trained Model URL]`

### 2. Generate Paraphrases
To generate paraphrases using PEARL generator, run:
```bash
bash paraphrase.sh
```

### 3. LLM Inference
To perform inference with the large language model, run:
```bash
bash inference.sh
```

---

## Citation
If you find this project helpful, please consider citing our work:

```bibtex
@inproceedings{fu-etal-2024-learning,
    title = "Learning to Paraphrase for Alignment with {LLM} Preference",
    author = "Fu, Junbo  and
      Zhao, Guoshuai  and
      Deng, Yimin  and
      Mi, Yunqi  and
      Qian, Xueming",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    pages = "2394--2407",
}
```

---

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![EMNLP 2024 Findings](https://img.shields.io/badge/EMNLP%202024-Findings-orange)](https://2024.emnlp.org/)

