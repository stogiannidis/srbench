# SRBench: Mind the Gap in Spatial Reasoning 🧠

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-stogian%2Fsrbench-blue.svg)](https://huggingface.co/datasets/stogian/srbench)
[![ArXiv](https://img.shields.io/badge/ArXiv-2503.19707-brown.svg)](https://arxiv.org/abs/2503.19707)

Welcome to **SRBench**, the ultimate test for Vision-Language Models (VLMs) in spatial reasoning! 🚀 This repository provides everything you need to evaluate, analyze, and push the boundaries of VLM capabilities.

Welcome to SRBench! This repository contains the source code for evaluating spatial reasoning in Vision-Language Models (VLMs). Below is an overview of the repository structure, installation, usage instructions, and contribution guidelines.

## Overview

SRBench is a comprehensive benchmarking suite for evaluating spatial reasoning capabilities in Vision-Language Models. The project includes:
- **Model Evaluation**: Scripts for evaluating various VLMs on spatial reasoning tasks
- **Data Processing**: Tools for creating and processing spatial reasoning datasets
- **Analysis Tools**: Utilities for computing accuracy and analyzing results
- **Flexible Evaluation**: Support for different prompting strategies (Chain-of-Thought, one-shot examples)

## Repository Structure

The project is organized as follows:
```
SRBench/
├── scripts/                              # Execution scripts
│   ├── run_closed.sh                   # Script to compute accuracy from results
│   └── run.sh                          # Main evaluation script
├── src/                                  # Source code of the project
│   ├── __init__.py                     # Package initialization
│   ├── eval.py                         # Main evaluation engine for VLMs
│   ├── eval_closed.py                  # Evaluation script for closed models
│   ├── data/                           # Data processing utilities
│   │   ├── __init__.py                 # Package initialization
│   │   ├── create_data.py              # Script for data creation
│   │   ├── create_images.py            # Script to create images
│   │   ├── create_prompts.py           # Script to generate prompts
│   │   ├── folding_pil.py              # PIL-based folding utilities
│   │   ├── images_dalle.py             # DALL-E image generation utilities
│   │   └── mrt.py                      # MRT utilities
│   ├── eval/                           # Evaluation utilities
│   │   ├── __init__.py                 # Package initialization
│   │   └── acc.py                      # Accuracy calculation utilities
│   └── utils/                          # Utility functions
│       ├── __init__.py                 # Package initialization
│       └── vlm/                        # VLM-specific utilities
│           ├── __init__.py             # Package initialization
│           ├── base.py                 # Base VLM classes
│           └── vlm_engine.py           # VLM engine implementation
├── .gitignore                          # Files and directories to ignore
├── requirements.txt                    # Required packages
├── LICENSE                             # MIT License file
└── README.md                           # Project documentation
```

## 🛠️ Installation

Get up and running in a few simple steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/stogiannidis/srbench.git
    cd srbench
    ```
2.  **Create a virtual environment:**
    Using `venv`:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    Or using `conda`:
    ```bash
    conda create -n srbench python=3.12
    conda activate srbench
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Evaluating Models

To evaluate a model on the SRBench dataset, use the following commands:

- **Standard Evaluation**:
  ```bash
  python src/eval.py --model <model_id> --dataset <dataset_id> --batch_size 16 --seed 42
  ```

- **Chain-of-Thought Prompting**:
  ```bash
  python src/eval.py --model <model_id> --dataset <dataset_id> --cot --batch_size 16
  ```

- **One-Shot Examples**:
  ```bash
  python src/eval.py --model <model_id> --dataset <dataset_id> --one-shot <path_to_example.json> --batch_size 16
  ```

### Running Batch Evaluations

The main evaluation script is configured in `scripts/run.sh`. You can run it with:
  ```bash
  bash scripts/run.sh
  ```
This script evaluates multiple models (InternVL, Qwen, LLaVA, etc.) on the SRBench dataset with Chain-of-Thought prompting.

### Computing Accuracy

To compute accuracy from evaluation results:
  ```bash
  bash scripts/run_closed.sh
  ```
This processes the CSV files in the output directory and calculates model accuracy.

### Command-line Arguments for `eval.py`

| Argument | Description | Default |
|---|---|---|
| `-m`, `--model` | Model identifier (required) | |
| `-d`, `--dataset` | Dataset identifier (required) | |
| `-b`, `--batch_size` | Batch size for processing | `16` |
| `--num_workers` | Number of data loading workers | `4` |
| `--max_samples` | Maximum samples to process | `None` |
| `--sample_strategy` | Sampling strategy (`first`, `random`, `stratified`) | `first` |
| `--device_map` | Device mapping strategy | `auto` |
| `--seed` | Random seed for reproducibility | `42` |
| `--cot` | Enable Chain-of-Thought prompting | |
| `--one-shot` | Path to one-shot example JSON file | |
| `-v`, `--verbose` | Enable verbose logging | |

<!-- ## 🖼️ Data Creation

The `src/data/` directory is your hub for crafting and shaping spatial reasoning datasets. Here's what you'll find:

-   `create_data.py`: The main script for generating datasets from JSON annotations.
-   `create_images.py`: A powerful tool for generating images.
-   `create_prompts.py`: Your go-to script for creating insightful prompts.
-   **Specialized Utilities**: A collection of scripts for handling various data formats.

## 🎯 Accuracy Calculation

The `src/eval/acc.py` script is designed to give you a clear picture of your model's performance. With it, you can:

-   **Extract Answers**: Use multiple strategies to pull answers from model responses.
-   **Calculate Accuracy**: Get exact match accuracy between predicted and gold answers.
-   **Analyze Splits**: Compute accuracy for different data splits.
-   **Generate Reports**: Create detailed reports of your evaluation results. -->

## Citation
```
@misc{stogiannidis2025mindgapbenchmarkingspatial,
      title={Mind the Gap: Benchmarking Spatial Reasoning in Vision-Language Models},
      author={Ilias Stogiannidis and Steven McDonagh and Sotirios A. Tsaftaris},
      year={2025},
      eprint={2503.19707},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19707},
}
```

## 🙌 Contributing

We welcome contributions from the community! To get involved, please follow these steps:

1.  **Fork the repository.**
2.  **Create a new branch** (`git checkout -b feature/your_feature`).
3.  **Commit your changes** (`git commit -am 'Add new feature'`).
4.  **Push to the branch** (`git push origin feature/your_feature`).
5.  **Open a Pull Request.**

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 📬 Contact

Have questions or feedback? Feel free to open an issue or reach out via email.
