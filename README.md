	# Repository Title

Welcome to our project! This repository contains all the source code, tests, and documentation required to understand and run the project. Below is an overview of the repository structure, installation, usage instructions, and contribution guidelines.

## Overview

This repository is divided into several modules that cover various aspects of the project, including:
- **Data Processing**: Scripts for loading, processing, and analyzing data.
- **Analysis Tools**: Modules that perform computations and run experiments.
- **Visualization Components**: Code for rendering results and generating reports.

## Repository Structure

The project is organized as follows:
```
/Root
 ├── bin/               # Storage for model binaries.
 ├── scripts/           # Bash scripts for running the project.
 ├── src/               # Source code of the project.
 │   ├── create_prompts.py  # Script to generate prompts.
 │   ├── create_images.py   # Script to create images.
 │	 └── test_vlms.py       # Test script for the VLMS model.
 ├── tests/             # Tests for the project.
 │   ├── test_pipelines.py  # Tests for pipeline.
 └── README.md          # This readme file.
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
