# GitHub Upload Instructions

## Browser Method

1. Sign in to GitHub.
2. Click the plus button in the upper-right corner.
3. Choose **New repository**.
4. Repository name: `EuroSAT-Capstone`
5. Description: `EuroSAT satellite image classification using PyTorch.`
6. Choose **Public** if the instructor needs to access it.
7. Do not create another README, `.gitignore`, or license.
8. Click **Create repository**.
9. Click **uploading an existing file**.
10. Open this project folder and upload its contents.
11. Commit message: `Add completed EuroSAT capstone project`
12. Click **Commit changes**.

Do not upload the downloaded `data` folder or large `.pth` model files through
the normal browser upload.

## PowerShell Method

Open PowerShell inside the project folder:

```powershell
git init
git add .
git commit -m "Add completed EuroSAT capstone project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/EuroSAT-Capstone.git
git push -u origin main
```
