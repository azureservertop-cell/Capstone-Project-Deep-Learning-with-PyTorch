from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report' / 'EuroSAT_Final_Report.pdf'
ASSETS = ROOT / '.tmp_report_assets'
ASSETS.mkdir(exist_ok=True)

classes = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']
counts = [3000,3000,3000,2500,2500,2000,2500,3000,2500,3000]

# Figure 1: class distribution
plt.figure(figsize=(10,4))
plt.bar(classes, counts)
plt.title('EuroSAT Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(ASSETS/'class_distribution.png', dpi=160)
plt.close()

# Figure 2: tuning validation loss
names = ['Experiment 1','Experiment 2','Experiment 3']
losses = [0.4675,0.4457,0.6175]
plt.figure(figsize=(8,4.5))
plt.bar(names, losses)
plt.ylabel('Best Validation Loss')
plt.title('Hyperparameter Tuning Results')
for i,v in enumerate(losses):
    plt.text(i, v+0.008, f'{v:.4f}', ha='center')
plt.ylim(0,0.7)
plt.tight_layout()
plt.savefig(ASSETS/'tuning_loss.png', dpi=160)
plt.close()

# Figure 3: baseline vs incremental final -- final wording intentionally uses "Baseline"
labels = ['Baseline','Incremental Final']
acc = [85.60,92.07]
f1 = [85.49,92.09]
x = np.arange(len(labels)); w = 0.35
plt.figure(figsize=(8.5,5.5))
plt.bar(x-w/2, acc, w, label='Accuracy')
plt.bar(x+w/2, f1, w, label='F1-score')
plt.ylabel('Score (%)')
plt.title('Baseline vs Incremental Training Setup')
plt.xticks(x, labels)
plt.ylim(0,100)
plt.legend()
plt.tight_layout()
plt.savefig(ASSETS/'pipeline_comparison.png', dpi=160)
plt.close()

# Figure 4: final model comparison
labels = ['Final 3-layer CNN','Frozen ResNet18']
acc = [92.07,82.44]
f1 = [92.09,82.32]
x = np.arange(len(labels)); w = 0.35
plt.figure(figsize=(8.5,5.5))
plt.bar(x-w/2, acc, w, label='Accuracy')
plt.bar(x+w/2, f1, w, label='F1-score')
plt.ylabel('Score (%)')
plt.title('Final Model Comparison')
plt.xticks(x, labels)
plt.ylim(0,100)
plt.legend()
plt.tight_layout()
plt.savefig(ASSETS/'model_comparison.png', dpi=160)
plt.close()

# Figure 5: final Custom CNN confusion matrix
cm = np.array([
[376,2,2,6,0,5,46,0,12,1],
[0,446,1,0,0,1,0,2,0,0],
[1,4,380,3,6,3,32,18,3,0],
[0,0,1,339,7,1,5,2,20,0],
[0,0,0,8,352,0,0,14,1,0],
[2,2,6,1,0,285,3,0,1,0],
[3,0,14,12,2,2,332,3,7,0],
[0,0,0,0,4,0,0,446,0,0],
[3,1,2,18,4,5,6,1,335,0],
[1,1,3,0,0,1,0,0,6,438]
])
plt.figure(figsize=(9,8))
plt.imshow(cm)
plt.title('Final Custom CNN Confusion Matrix')
plt.xlabel('Predicted label'); plt.ylabel('True label')
plt.xticks(np.arange(10), classes, rotation=45, ha='right')
plt.yticks(np.arange(10), classes)
for i in range(10):
    for j in range(10):
        plt.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='yellow' if cm[i,j] < 250 else 'black')
plt.colorbar()
plt.tight_layout()
plt.savefig(ASSETS/'incremental_confusion.png', dpi=160)
plt.close()

# Figure 6: ablation results
abl_labels = ['Incremental Baseline','+ ColorJitter','+ VerticalFlip','+ Rotation pipeline']
abl_acc = [92.07,91.48,89.83,67.60]
plt.figure(figsize=(8.5,5.2))
plt.bar(abl_labels, abl_acc)
plt.ylabel('Test Accuracy (%)')
plt.title('Data Augmentation Ablation Study')
plt.ylim(60,95)
plt.xticks(rotation=15, ha='right')
for i,v in enumerate(abl_acc):
    plt.text(i, v+0.4, f'{v:.2f}%', ha='center')
plt.tight_layout()
plt.savefig(ASSETS/'ablation_accuracy.png', dpi=160)
plt.close()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=22, leading=28, spaceAfter=18))
styles.add(ParagraphStyle(name='SubTitleCenter', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, leading=16, textColor=colors.HexColor('#4b5563'), spaceAfter=8))
styles.add(ParagraphStyle(name='H1Custom', parent=styles['Heading1'], fontSize=16, leading=20, spaceBefore=12, spaceAfter=8, textColor=colors.HexColor('#111827')))
styles.add(ParagraphStyle(name='BodyCustom', parent=styles['BodyText'], fontSize=10.5, leading=14, spaceAfter=6))
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=9, leading=12, textColor=colors.HexColor('#4b5563')))

page_w, page_h = letter
margin = 0.65 * inch
content_w = page_w - 2 * margin

def img(name, width=content_w, max_h=3.8*inch):
    p = ASSETS / name
    im = Image(str(p))
    iw, ih = im.imageWidth, im.imageHeight
    scale = min(width/iw, max_h/ih)
    im.drawWidth = iw*scale; im.drawHeight = ih*scale; im.hAlign = 'CENTER'
    return im

def t(data, widths=None, fontsize=9):
    if widths is None:
        widths = [content_w/len(data[0])] * len(data[0])
    tbl = Table(data, colWidths=widths, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#111827')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),fontsize),
        ('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f9fafb')]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    return tbl

story=[]
story += [Spacer(1,1.1*inch), Paragraph('EuroSAT Satellite Image Classification',styles['TitleCenter']), Paragraph('Deep Learning Capstone Report - Final Experiment Version',styles['SubTitleCenter']), Paragraph('Jianye Chen<br/>Fanshawe College<br/>PyTorch Image Classification Project',styles['SubTitleCenter']), Spacer(1,0.4*inch)]
story.append(t([['Final Model','Test Accuracy','Weighted F1-score'],['3-layer Custom CNN','92.07%','92.09%'],['Frozen ResNet18','82.44%','82.32%']], widths=[content_w*.45,content_w*.275,content_w*.275], fontsize=10))
story += [Spacer(1,.4*inch), Paragraph('Final choice: simple augmentation with restricted RandomResizedCrop and HorizontalFlip. VerticalFlip, Rotation pipeline, and ColorJitter were tested but not adopted because they did not improve the final model.',styles['BodyCustom']), PageBreak()]

story += [Paragraph('1. Project Overview',styles['H1Custom']), Paragraph('The objective of this capstone project is to build, train, evaluate, and compare deep learning models for image classification. The selected problem is satellite image land-use classification using the EuroSAT RGB dataset. The project uses PyTorch with standard image-classification components: Dataset/DataLoader, CNN layers, BatchNorm, Dropout, CrossEntropyLoss, Adam optimization, validation-based early stopping, and a frozen ResNet18 transfer-learning comparison.',styles['BodyCustom']), Paragraph('The project compares a custom 3-layer convolutional neural network trained from scratch with a pretrained ResNet18 model used as a fixed feature extractor.',styles['BodyCustom'])]
story += [Paragraph('2. Dataset',styles['H1Custom']), Paragraph('EuroSAT contains 27,000 RGB satellite images across 10 land-use classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake. A stratified split was used to preserve class proportions across the training, validation, and test sets.',styles['BodyCustom'])]
story.append(t([['Split','Number of Images','Purpose'],['Training','18,900','Model parameter learning'],['Validation','4,050','Hyperparameter selection and early stopping'],['Test','4,050','Final generalization evaluation']], widths=[content_w*.25,content_w*.25,content_w*.50]))
story += [Spacer(1,.15*inch), img('class_distribution.png',max_h=2.8*inch), Paragraph('Figure 1. EuroSAT class distribution.',styles['Small']), PageBreak()]

story += [Paragraph('3. Preprocessing and Augmentation',styles['H1Custom']), Paragraph('All images were resized to 64 x 64 and normalized using ImageNet-style mean and standard deviation values. The final training transform intentionally stays simple: a restricted RandomResizedCrop with scale=(0.8, 1.0), RandomHorizontalFlip, ToTensor, and Normalize.',styles['BodyCustom']), Paragraph('The restricted crop is important because satellite land-use labels often depend on global spatial structure such as field boundaries, road shape, river direction, or residential layout. A default random crop may remove too much of this structure.',styles['BodyCustom'])]
story.append(t([['Transform','Used in Final Model?','Reason'],['RandomResizedCrop(scale=(0.8, 1.0))','Yes','Preserves most of the satellite image while adding variation'],['RandomHorizontalFlip','Yes','Simple orientation variation'],['RandomVerticalFlip','No','Ablation decreased test accuracy'],['RandomRotation pipeline','No','Tested pipeline performed poorly and added extra preprocessing effects'],['ColorJitter','No','No clear improvement over baseline']], widths=[content_w*.33,content_w*.18,content_w*.49], fontsize=8.5))
story += [Paragraph('4. Custom CNN Architecture',styles['H1Custom']), Paragraph('The custom model is a 3-layer CNN with three convolution blocks. Each block uses Conv2d, BatchNorm, ReLU, and MaxPool. After flattening, a fully connected layer with Dropout is used before the final 10-class output layer.',styles['BodyCustom'])]
story.append(t([['Stage','Operation'],['Block 1','Conv2d(3->32), BatchNorm, ReLU, MaxPool'],['Block 2','Conv2d(32->64), BatchNorm, ReLU, MaxPool'],['Block 3','Conv2d(64->128), BatchNorm, ReLU, MaxPool'],['Classifier','Flatten -> Linear(128*8*8 -> 256) -> Dropout -> Linear(256 -> 10)']], widths=[content_w*.25,content_w*.75]))
story.append(PageBreak())

story += [Paragraph('5. Training Strategy',styles['H1Custom']), Paragraph('The model was trained using CrossEntropyLoss and Adam. Validation loss was monitored at the end of each epoch. When validation loss improved, the model state_dict was saved. If validation loss failed to improve for several epochs, early stopping was triggered and the best saved model was reloaded.',styles['BodyCustom']), Paragraph('This prevents the final test result from depending on a later epoch that may have overfit the training data.',styles['BodyCustom'])]
story += [Paragraph('6. Hyperparameter Tuning',styles['H1Custom']), Paragraph('Three manual hyperparameter settings were compared. The final setting was selected using the lowest validation loss, not the highest training accuracy.',styles['BodyCustom'])]
story.append(t([['Experiment','Learning Rate','Batch Size','Dropout','Best Val Loss','Val Acc at Best Loss'],['Experiment 1','0.0010','32','0.30','0.4675','83.51%'],['Experiment 2','0.0005','32','0.30','0.4457','84.12%'],['Experiment 3','0.0010','64','0.50','0.6175','78.54%']], widths=[content_w*.23,content_w*.16,content_w*.14,content_w*.14,content_w*.16,content_w*.17], fontsize=8.5))
story += [Spacer(1,.15*inch), img('tuning_loss.png',max_h=3*inch), Paragraph('Figure 2. Hyperparameter tuning by validation loss. Experiment 2 was selected.',styles['Small']), PageBreak()]

story += [Paragraph('7. Baseline vs Incremental Improvement',styles['H1Custom']), Paragraph('The first baseline used a simple 3-layer CNN but only four tuning epochs and the default RandomResizedCrop. It achieved 85.60% test accuracy. The incremental version kept the same CNN architecture but changed only two training details: restricted crop scale and six tuning epochs. This improved test accuracy to 92.07%.',styles['BodyCustom'])]
story.append(t([['Version','Validation Accuracy','Test Accuracy','Precision','Recall','F1-score'],['Baseline','84.37%','85.60%','85.89%','85.60%','85.49%'],['Incremental Final CNN','90.96%','92.07%','92.37%','92.07%','92.09%']], widths=[content_w*.28,content_w*.16,content_w*.14,content_w*.14,content_w*.14,content_w*.14], fontsize=8.5))
story += [Spacer(1,.1*inch), img('pipeline_comparison.png',max_h=3.1*inch), Paragraph('Figure 3. The improved training setup recovered strong performance without increasing architecture complexity.',styles['Small']), PageBreak()]

story += [Paragraph('8. Final Model Results',styles['H1Custom']), Paragraph('The final selected model is the incremental 3-layer CNN with learning rate 0.0005, batch size 32, dropout 0.3, restricted random crop, and horizontal flipping. It achieved strong test-set generalization.',styles['BodyCustom'])]
story.append(t([['Model','Validation Accuracy','Test Accuracy','Precision','Recall','F1-score'],['Final 3-layer CNN','90.96%','92.07%','92.37%','92.07%','92.09%'],['Frozen ResNet18','81.53%','82.44%','82.93%','82.44%','82.32%']], widths=[content_w*.27,content_w*.16,content_w*.14,content_w*.14,content_w*.14,content_w*.15], fontsize=8.5))
story += [Spacer(1,.1*inch), img('model_comparison.png',max_h=3.1*inch), Paragraph('Figure 4. Final Custom CNN vs frozen ResNet18.',styles['Small']), PageBreak()]

story += [Paragraph('9. Confusion Matrix and Misclassification Analysis',styles['H1Custom']), Paragraph('The confusion matrix shows that the final model performs strongly across most classes. Remaining errors are mainly between visually similar land-use categories, such as agricultural and vegetation classes or linear structures such as rivers and highways.',styles['BodyCustom']), img('incremental_confusion.png',max_h=5.4*inch), Paragraph('Figure 5. Final Custom CNN confusion matrix.',styles['Small']), PageBreak()]

story += [Paragraph('10. Ablation Study: Why Not Use Extra Augmentation?',styles['H1Custom']), Paragraph('Additional augmentation methods were tested one at a time while keeping the selected CNN architecture and hyperparameters fixed. None of the tested methods improved the final baseline.',styles['BodyCustom'])]
story.append(t([['Augmentation Setting','Test Accuracy','Weighted F1','Accuracy Change'],['Incremental Baseline','92.07%','92.09%','0.00 pp'],['+ ColorJitter','91.48%','91.49%','-0.59 pp'],['+ VerticalFlip','89.83%','89.95%','-2.25 pp'],['+ Rotation pipeline','67.60%','67.46%','-24.47 pp']], widths=[content_w*.38,content_w*.20,content_w*.20,content_w*.22], fontsize=8.5))
story += [Spacer(1,.1*inch), img('ablation_accuracy.png',max_h=3*inch), Paragraph('Figure 6. Additional augmentation did not improve the final model.',styles['Small']), Paragraph('The rotation result is interpreted cautiously because the tested rotation pipeline also introduced resize-and-crop operations to avoid black borders. Therefore, the large decrease is not claimed to be the isolated effect of rotation alone. It only shows that this tested rotation preprocessing pipeline was not suitable for the final model.',styles['BodyCustom']), PageBreak()]

story += [Paragraph('11. Potential Problems and Mitigations',styles['H1Custom'])]
story.append(t([['Potential Issue','Mitigation Used'],['Data leakage','Train, validation, and test sets were separated before model training. The test set was only used for final evaluation.'],['Overfitting','Dropout, BatchNorm, data augmentation, validation monitoring, and early stopping were used.'],['Misleading overall accuracy','Precision, recall, F1-score, and confusion matrices were also evaluated.'],['Overly aggressive augmentation','Ablation testing showed that more augmentation was not automatically better. The final model kept only the simple augmentation that performed best.'],['Transfer-learning domain difference','ResNet18 was treated as a frozen-feature baseline. Future work could fine-tune later layers for satellite images.']], widths=[content_w*.30,content_w*.70], fontsize=8.5))
story += [Paragraph('12. Conclusion',styles['H1Custom']), Paragraph('This project built and evaluated a full PyTorch image-classification pipeline for EuroSAT. The final 3-layer CNN achieved 92.07% test accuracy and 92.09% weighted F1-score, outperforming the frozen ResNet18 comparison. The most important finding was that performance improved substantially by adjusting the training setup instead of increasing network complexity.',styles['BodyCustom']), Paragraph('The ablation study also showed that extra preprocessing should not be added automatically. VerticalFlip, ColorJitter, and the tested rotation pipeline did not improve the final model, so the final solution uses the simplest tested augmentation pipeline that achieved the best result.',styles['BodyCustom']), Paragraph('Future work could include fine-tuning the last ResNet blocks, testing a larger image size, using Grad-CAM to interpret the CNN decision process, and running multiple random seeds to measure training variability.',styles['BodyCustom'])]

def footer(canvas,doc):
    canvas.saveState(); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#6b7280'))
    canvas.drawString(margin,.35*inch,'EuroSAT Satellite Image Classification - Capstone Report')
    canvas.drawRightString(page_w-margin,.35*inch,f'Page {doc.page}')
    canvas.restoreState()

pdf=SimpleDocTemplate(str(OUT),pagesize=letter,rightMargin=margin,leftMargin=margin,topMargin=.6*inch,bottomMargin=.6*inch)
pdf.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
