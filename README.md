# Pill Classification on Low Quality Image

## Objective
We aim to build a system to predict the name of the pill from an input image. There are **15 types of pills** in our dataset. Each input image contains exactly one pill, but images may be **partially occluded, blurred, noisy, or affected by glare** – posing significant challenges for classification. Therefore, we need a robust AI solution that generalizes well and accurately predicts the pill type under a wide range of conditions, from low-quality to clear images.

## Solution
We propose a **two‑stage system**:

1. **Pill detection (localization)** – using YOLO11 fine‑tuned with Retinex and other augmentations.
2. **Pill classification** – using a **ResNet18** (required by our project) trained with **cut‑mix** and **mix‑up** augmentations.

Our system achieved a **macro F1 score of 0.93** on the leaderboard of the **VAIC contest** (Intro2AI course, Hanoi University of Science and Technology).  
🔗 [Contest link](https://vaic.aidemyx.edu.vn/competitions/20)

> **Note:** The data in this repository is already preprocessed for new programmers.

## Instructions
Follow these steps to reproduce our pipeline:

1. **Install necessary packages**  
   Clone this repository, open a terminal, and install missing packages.  
   If your machine has a GPU, install `torch` and `torchvision` with CUDA support.

2. **Fine‑tune YOLO**  
   ```bash
   python album.py
3. **Detect pills**
First run detection on the training dataset, then change the directory to the test dataset.

    ```bash
    python detect.py
4. **Crop and resize**
Crop the detected pills and resize them for the classification stage (both train and test sets).

    ```bash
    python crop.py
5. **Train the classification model**
Open full_pipeline.ipynb in Jupyter Notebook.
You don’t have to run the first two cells of this notebook.

6. **Obtain predictions**
The final predictions for the test files will be in the cropped_224 folder 😅

## Notice
This project was created as part of our third‑semester university coursework. There may be mistakes such as:

Lack of optimality

Poor folder management

Missing files (we had to delete thousands of files and might have accidentally thrown away something important)

Please adjust file/directory paths if you encounter any issues.
The visualized_result/ folder contains example images with bounding boxes – helpful for beginners to understand what a bounding box looks like.
