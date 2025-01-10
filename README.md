# Pneumonia Classification Using Chest X-Ray Images

DTU Machine Learning Operations

## Team Members
- **Lukas Korinek** (s246710)  
- **Frederik Sartov Olsen** (s204118)  
- **Konstantinos-Athanasios Papagoras PhD** (s230068)  
- **Yessin Moakher** (s250283)  

---

## The overall goal of the project

The primary goal of this project is to classify pneumonia using chest X-ray images.

---

## What framework are you going to use, and how do you intend to include the framework into your project?

We will use the PyTorch Image Models (timm) framework for this project.

---

## What data are you going to run on (initially, may change).


This project aims to classify chest X-ray images into two categories: Pneumonia and Normal, using the Kaggle dataset  ["Chest X-Ray Images (Pneumonia)"](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The dataset contains 5,863 pediatric chest X-rays from Guangzhou Women and Children’s Medical Center, all captured as part of routine clinical care and graded by expert physicians. For the initial stages of the project, we will use a subset of the dataset to verify that everything is running smoothly, before scaling up to the full dataset. The dataset is split into training, validation, and testing folders, making it ideal for this task. It was chosen for its simplicity and suitability for beginner-level image classification projects, especially in healthcare, and it seems feasible to implement within a short timeframe.

---

## What models do you expect to use

For this project, we will begin with a baseline model using a simple convolutional neural network (CNN) to establish a reference performance. We will then leverage pre-trained models from PyTorch’s image models framework, such as ResNet50, VGG16, and DenseNet, to improve classification accuracy. These models will be fine-tuned for our specific task by adapting the final layers to classify X-ray images into two categories: Pneumonia and Normal. We will use torchvision for accessing pre-trained models and data augmentation, and torch.optim for optimization, evaluating the models based on accuracy, precision, recall, and F1-score.

---

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
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
