# SRBench

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-stogiannidis%2Fsrbench-blue.svg)](https://huggingface.co/datasets/stogian/srbench)

Welcome to our project! This repository contains all the source code, tests, and documentation required to understand and run the project. Below is an overview of the repository structure, installation, usage instructions, and contribution guidelines.

## Overview

This repository is divided into several modules that cover various aspects of the project, including:
- **Data Processing**: Scripts for loading, processing, and analyzing data.
- **Analysis Tools**: Modules that perform computations and run experiments.
- **Visualization Components**: Code for rendering results and generating reports.

## Repository Structure

The project is organized as follows:
```
SRBench/
├── bin/                                  # Storage for model binaries
├── scripts/                              # Bash scripts for running the project
│   └── run.sh                            # Main execution script
├── src/                                  # Source code of the project
│   ├── data_creation/                    # Scripts for data creation
│       ├── __init__.py                	  # Initialization file
│       ├── create_data.py                # Script for data creation
│       ├── create_images.py              # Script to create images
│       └── create_prompts.py             # Script to generate prompts
│   ├── utils/                            # Utility functions
│       ├── __init__.py                   # Initialization file
│       ├── vlm_helpers.py                # Helper functions for the VLM models
│   ├── eval.py                           # Evaluation script
│   ├── eval_intern.py 				      # Evaluation script for InternVL
│   ├── eval_openai.py 				      # Evaluation script for OpenAI models
│   ├── eval_mini.py 				      # Evaluation script for MiniCPM-V
├── .gitignore                            # Files and directories to ignore
├── requirements.txt                      # Required packages
├── LICENSE                               # MIT License file
└── README.md                             # Project documentation
```

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/stogiannidis/srbench.git
	cd srbench
	```
2. Create a virtual environment:
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```
	or using `conda`:
	```bash
	conda create -n srbench python=3.12
	conda activate srbench
	```
3. Install the required packages:
	```bash
	pip install -r requirements.txt
	```

## Usage

To run the project, follow these steps:
1. Fetch the dataset from `Hugging Face`:
	```bash
	huggingface-cli login
	huggingface-cli download stogiannidis/srbench
	```
2. Run the script:
	```bash
	bash scripts/run.sh
	```

> **Note**: This is a TODO – dataset is not available yet.

## Contributing

Contributions are welcome! Please follow these steps:
- Fork the repository.
- Create a new branch (`git checkout -b feature/your_feature`).
- Commit your changes (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature/your_feature`).
- Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For questions or feedback, please open an issue or contact me via email.
