# Report

Use the final project results from the updated notebook and root README when preparing the PDF report.

Final selected model:

- 3-layer Custom CNN
- Test accuracy: **92.07%**
- Weighted F1-score: **92.09%**
- Frozen ResNet18 test accuracy: **82.44%**

Final augmentation choice: `RandomResizedCrop(scale=(0.8,1.0))` + `RandomHorizontalFlip`. VerticalFlip, Rotation, and ColorJitter were tested but not retained because none improved the final baseline.
