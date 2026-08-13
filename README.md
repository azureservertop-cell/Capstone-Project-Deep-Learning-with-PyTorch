# EuroSAT Satellite Image Classification

A PyTorch capstone project that classifies EuroSAT RGB satellite images into 10 land-use categories using deep learning methods.

## Project Overview

The final project compares:

1. A **3-layer custom CNN** trained from scratch
2. A pretrained **ResNet18** used as a frozen feature extractor

The workflow includes stratified train/validation/test splitting, simple data augmentation, manual hyperparameter comparison, validation-loss early stopping, test-set evaluation, confusion-matrix analysis, and an augmentation ablation study.

## Dataset

EuroSAT contains **27,000 RGB satellite images** across 10 classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Split used in the project:

- Training: **70% (18,900 images)**
- Validation: **15% (4,050 images)**
- Test: **15% (4,050 images)**

## Final Custom CNN

The final CNN uses a simple convolutional architecture:

- Conv2d: 3 → 32
- Conv2d: 32 → 64
- Conv2d: 64 → 128
- BatchNorm + ReLU + MaxPool
- Flatten
- Fully connected layer: 8192 → 256
- Dropout
- Output layer: 256 → 10

### Final Training Transform

The best-performing augmentation pipeline was deliberately kept simple:

```python
transforms.RandomResizedCrop(
    64,
    scale=(0.8, 1.0)
),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean, std)
```

`RandomVerticalFlip`, `RandomRotation`, and `ColorJitter` were tested but were **not retained** because none improved the final baseline.

## Hyperparameter Tuning

Three manual settings were compared for **6 epochs**. The final setting was selected using the **lowest validation loss**.

| Experiment | Learning Rate | Batch Size | Dropout | Best Validation Loss | Validation Accuracy at Best Loss |
|---|---:|---:|---:|---:|---:|
| **Experiment 2** | **0.0005** | **32** | **0.30** | **0.4457** | **84.12%** |
| Experiment 1 | 0.0010 | 32 | 0.30 | 0.4675 | 83.51% |
| Experiment 3 | 0.0010 | 64 | 0.50 | 0.6175 | 78.54% |

Experiment 2 was selected for the final CNN.

## Final Results

| Model | Validation Accuracy | Test Accuracy | Precision | Recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **Final 3-layer Custom CNN** | **90.96%** | **92.07%** | **92.37%** | **92.07%** | **92.09%** |
| Frozen ResNet18 | 81.53% | 82.44% | 82.93% | 82.44% | 82.32% |

The custom CNN achieved the best overall performance in the final controlled run.

## Development Finding

An earlier baseline run used the same 3-layer CNN but the default `RandomResizedCrop` and only **4 tuning epochs**. In the controlled comparison it reached **85.60%** test accuracy.

Restricting the crop to `scale=(0.8, 1.0)` and increasing tuning to **6 epochs** improved the test result to **92.07%** without increasing the CNN depth.

This suggests that the earlier performance loss was mainly related to the training setup rather than the simplified 3-layer architecture.

## Augmentation Ablation Study

| Augmentation Setting | Test Accuracy | Weighted F1 | Accuracy Change |
|---|---:|---:|---:|
| **Final baseline: crop + horizontal flip** | **92.07%** | **92.09%** | **0.00 pp** |
| + Color Jitter | 91.48% | 91.49% | -0.59 pp |
| + Vertical Flip | 89.83% | 89.95% | -2.25 pp |
| + Rotation pipeline | 67.60% | 67.46% | -24.47 pp |

The tested rotation pipeline also included resize-and-center-crop operations to avoid black borders, so its large decrease should **not** be interpreted as the isolated effect of rotation itself. It only shows that the tested rotation preprocessing pipeline was not suitable.

Because no additional augmentation outperformed the simpler baseline, the final model does **not** use VerticalFlip, Rotation, or ColorJitter.

## ResNet18 Comparison

ResNet18 was pretrained on ImageNet. Its pretrained feature parameters were frozen and only the final classification layer was trained for the 10 EuroSAT classes.

The final ResNet18 test accuracy was **82.44%**. This result applies to the frozen-feature setup and should not be interpreted as evidence that transfer learning is generally worse than a custom CNN. Fine-tuning some later ResNet layers could be explored in future work.

## Project Structure

```text
Capstone-Project-Deep-Learning-with-PyTorch/
├── EuroSAT_Capstone_Project.ipynb
├── README.md
├── requirements.txt
├── results/
│   ├── hyperparameter_results.csv
│   ├── model_comparison.csv
│   └── ablation_results.csv
├── report/
│   └── EuroSAT_Final_Report.pdf
├── presentation/
│   └── EuroSAT_Capstone_Presentation.pptx
├── images/
└── docs/
```

## Running the Notebook

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `EuroSAT_Capstone_Project.ipynb` and run the cells from top to bottom.

The notebook contains the final code and recorded result summaries used for the project report.

## Final Conclusion

The final solution uses a **3-layer Custom CNN**, learning rate **0.0005**, batch size **32**, dropout **0.3**, restricted random crop, horizontal flipping, BatchNorm, and early stopping.

The final held-out test accuracy is **92.07%** with a weighted F1-score of **92.09%**. The experiments also show that more preprocessing was not automatically better, so the project retains the simplest tested augmentation strategy that achieved the strongest result.

## Author

**Name:** Jianye Chen  
**College:** Fanshawe College  
**Course:** Deep Learning Capstone
