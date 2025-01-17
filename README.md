# Pneumonia Classification Using Chest X-Ray Images

DTU Machine Learning Operations

## Team Members

- **Lukas Korinek** (s246710)
- **Frederik Sartov Olsen** (s204118)
- **Konstantinos-Athanasios Papagoras PhD** (s230068)
- **Yessin Moakher** (s250283)

---

## Project Goal

The primary goal of this project is to classify pneumonia using chest X-ray images.

---

## Framework and Usage

We will use the PyTorch Image Models (timm) framework for this project.

---

## Dataset Information

This project aims to classify chest X-ray images into two categories: Pneumonia and Normal, using the Kaggle dataset  ["Chest X-Ray Images (Pneumonia)"](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The dataset contains 5,863 pediatric chest X-rays from Guangzhou Women and Children’s Medical Center, all captured as part of routine clinical care and graded by expert physicians. For the initial stages of the project, we will use a subset of the dataset to verify that everything is running smoothly, before scaling up to the full dataset. The dataset is split into training, validation, and testing folders, making it ideal for this task. It was chosen for its simplicity and suitability for beginner-level image classification projects, especially in healthcare, and it seems feasible to implement within a short timeframe.

---

## Models

For this project, we will begin with a baseline model using a simple convolutional neural network (CNN) to establish a reference performance. We will then leverage pre-trained models from PyTorch’s image models framework, such as ResNet50, VGG16, and DenseNet, to improve classification accuracy. These models will be fine-tuned for our specific task by adapting the final layers to classify X-ray images into two categories: Pneumonia and Normal. We will use torchvision for accessing pre-trained models and data augmentation, and torch.optim for optimization, evaluating the models based on accuracy, precision, recall, and F1-score.

---

## Automation Tasks

This `tasks.py` file contains automation tasks. It uses the `invoke` library.

### Prerequisites

- Install `invoke`:
  ```bash
  pip install invoke
  ```
- Create a `.env` file with required environment variables (e.g., `WANDB_API_KEY`).

### Commands

#### Setup Tasks

1. **Create Environment**

   ```bash
   invoke create-environment
   ```

   Creates a new Conda environment for the project (don't forget to activate it before installing requirements).

2. **Install Requirements**

   ```bash
   invoke requirements
   ```

   Installs the project dependencies from `requirements.txt` and local `pip` configuration.

3. **Install Development Requirements**

   ```bash
   invoke dev-requirements
   ```

   Installs development dependencies.

#### Core Tasks

4. **Preprocess Data**

   ```bash
   invoke preprocess-data --percentage=<float>
   ```

   Preprocesses raw data and stores it in the processed directory. Use the `--percentage` argument to specify a fraction of data to process (default is `1.0`).

5. **Train Model**

   ```bash
   invoke train
   ```

   Executes the model training script.

6. **Run Tests**

   ```bash
   invoke test
   ```

   Runs tests using `pytest` and generates a coverage report.

7. **Test Coverage Report**

   ```bash
   invoke test-coverage
   ```

   Runs tests and displays a detailed coverage report.

#### Docker Tasks

8. **Build Docker Image**

   ```bash
   invoke docker-build --progress=<plain|auto>
   ```

   Builds the Docker image for the project using the specified Dockerfile.

9. **Run Docker Training**

   ```bash
   invoke docker-train
   ```

   Runs the training process in a Docker container. Requires `WANDB_API_KEY` in the environment.

#### W&B Tasks

10. **Run W&B Sweep**
    ```bash
    invoke wandb-sweep --config-path=<path>
    ```
    Creates a Weights & Biases sweep from the specified config file and programmatically starts the agent. The default configuration path is `configs/sweep.yaml`. You can specify a different path using the `--config-path` argument if needed.

#### Formatting Tasks

11. **Format Code with Ruff**
    ```bash
    invoke ruff-format
    ```
    Formats the project files using `ruff`.

#### Docs Tasks

12. **Build Documentation**

    ```bash
    invoke build-docs
    ```

    Builds the project documentation using `mkdocs`.

13. **Serve Documentation**

    ```bash
    invoke serve-docs
    ```

    Serves the project documentation locally for preview.

### Notes

- Ensure all required tools and dependencies are properly installed before running tasks.
- For additional details, refer to the `tasks.py` source code.

---

## Directory Structure

The directory structure of the project looks like this:

```txt
├── .github/                  # GitHub actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```
