# Presentation

Use the final project results from the updated notebook and root README when preparing the presentation.

Key presentation numbers:

- Final 3-layer Custom CNN test accuracy: **92.07%**
- Weighted F1-score: **92.09%**
- Frozen ResNet18 test accuracy: **82.44%**
- Initial controlled baseline: **85.60%**
- Incremental improvement: **+6.47 percentage points**

Ablation finding: additional ColorJitter, VerticalFlip, and the tested Rotation pipeline did not outperform the simpler final augmentation. The final pipeline therefore keeps only the restricted random crop and horizontal flip.

Final presentation file: `EuroSAT_Capstone_Presentation.pptx`
