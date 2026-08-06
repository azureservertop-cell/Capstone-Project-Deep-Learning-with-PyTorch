# EuroSAT Satellite Image Classification

A PyTorch deep learning capstone project for classifying satellite images into 10 land-use categories.

## Project Overview

This project builds, trains, evaluates, and compares deep learning models using the EuroSAT dataset.

Main work:

- Image resizing and normalization
- Data augmentation
- Stratified train, validation, and test split
- Custom CNN
- Hyperparameter tuning
- Early stopping
- ResNet18 transfer learning
- Accuracy, precision, recall, and F1-score
- Confusion matrix
- Correct and incorrect prediction examples

## Dataset

The EuroSAT RGB dataset contains 27,000 satellite images in 10 classes:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

Data split:

- Training: 70%
- Validation: 15%
- Test: 15%

## Models

### Custom CNN

The custom model contains four convolution blocks, batch normalization, ReLU activation,
max pooling, adaptive average pooling, dropout, and fully connected layers.

### ResNet18

A pretrained ResNet18 is used for transfer learning. The feature layers are frozen and
the final layer is replaced with a 10-class output layer.

## Hyperparameter Experiments

| Experiment | Learning Rate | Batch Size | Dropout |
|---|---:|---:|---:|
| Experiment 1 | 0.001 | 32 | 0.30 |
| Experiment 2 | 0.0005 | 32 | 0.30 |
| Experiment 3 | 0.001 | 64 | 0.50 |

## Results

Replace these values after running the complete notebook.

| Model | Validation Accuracy | Test Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Custom CNN | TBD | TBD | TBD | TBD | TBD |
| ResNet18 | TBD | TBD | TBD | TBD | TBD |

### Main Findings

- Best model: `TBD`
- Best test accuracy: `TBD`
- Best F1-score: `TBD`
- Most difficult classes: `TBD`

## Project Structure

```text
EuroSAT-Capstone/
├── EuroSAT_Capstone_Project.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── GITHUB_UPLOAD_GUIDE.md
├── results/
├── images/
├── report/
├── presentation/
└── docs/
    └── index.html
```

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd EuroSAT-Capstone
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `EuroSAT_Capstone_Project.ipynb`.

For a quick check:

```python
FAST_MODE = True
```

For final results:

```python
FAST_MODE = False
```

## Evaluation

The models are evaluated using accuracy, precision, recall, F1-score,
classification reports, confusion matrices, and prediction examples.

## Future Improvements

- Try a larger image size
- Unfreeze more ResNet18 layers
- Test ResNet34 or EfficientNet
- Add more hyperparameter experiments
- Add Grad-CAM visualization

## Author

**Name:** Jianye Chen  
**Course:** ____________________  
**College:** Fanshawe College  
**Date:** ____________________
