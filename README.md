# 🐶🐱 Cat vs Dog Image Classification using SVM

This project focuses on building a binary image classifier to distinguish between dog and cat images using Support Vector Machine (SVM) from scikit-learn.

## 🧠 Problem Statement

Classifying pet images is a common computer vision task. However, traditional ML models like SVM often struggle with image pixel data. The aim here is to evaluate how an SVM model performs in a raw image classification task and learn its limitations and strengths.

---

 ✅ Solution Approach

🔹 Step 1: Dataset Loading

We used a structured folder setup:
Task 3/
├── images/
│ ├── dog/
│ └── cat/


Each image was read using OpenCV, resized to `64x64`, and flattened to a 1D array.

---

🔹 Step 2: Preprocessing

- **Image Resize** to reduce complexity.
- **Flatten** each image to feed into the SVM.
- **Label Encode** the string labels using `LabelEncoder`.

---

🔹 Step 3: Model Training

Used `sklearn.svm.SVC` with default parameters.

```python
model = SVC()
model.fit(X_train, y_train)
```

🔹 Step 4: Evaluation
Used:

accuracy_score to get overall accuracy

classification_report for detailed precision/recall/f1

📊 Results:
Accuracy: 52.5%
Precision (cat): 0.39 | Recall: 0.48 | F1-score: 0.43
Precision (dog): 0.64 | Recall: 0.55 | F1-score: 0.59

📉 Observations
SVM performed better on dog images than cat.

The model has limited capacity to understand pixel relationships.

The flattening process loses spatial features.

🔄 Improvements to Try
Use PCA for dimensionality reduction before SVM.

Extract HOG features instead of raw pixels.

Switch to CNN-based models for superior performance.

Apply data augmentation to balance learning.

👤 Author: Krrish Yaduka
📧 Contact: "kyaduka2803@gmail.com"
Linkedin : "linkedin.com/in/krrish-yaduka-16aa72293/"
