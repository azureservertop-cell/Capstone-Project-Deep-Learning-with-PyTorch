# Upload the Project to GitHub

## Using the GitHub Website

1. Sign in to GitHub.
2. Click the **+** button and choose **New repository**.
3. Repository name: `EuroSAT-Capstone`
4. Description: `Satellite image classification using a custom CNN and ResNet18 in PyTorch.`
5. Select **Public**.
6. Do not create another README.
7. Click **Create repository**.
8. Select **uploading an existing file**.
9. Upload all files and folders from this package.
10. Commit message: `Initial EuroSAT capstone project`
11. Click **Commit changes**.

## Using PowerShell

```powershell
git init
git add .
git commit -m "Initial EuroSAT capstone project"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

## Optional GitHub Pages

1. Open repository **Settings**.
2. Select **Pages**.
3. Choose **Deploy from a branch**.
4. Select branch `main`.
5. Select folder `/docs`.
6. Save.
