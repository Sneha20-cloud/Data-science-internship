#  Task 1: Iris Flower Classification

An end-to-end Machine Learning project that classifies Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — using four flower measurements.

---

##  Objective
Train and evaluate ML models to classify iris species based on:
- Sepal Length & Width
- Petal Length & Width

---

## 📂 Project Structure
```
iris_classification/
│
├── iris_classification.py     # Main ML script (all steps)
├── iris_eda.png               # Exploratory Data Analysis plots
├── iris_confusion_matrices.png# Confusion matrices for all models
├── iris_model_comparison.png  # Model accuracy comparison chart
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🧠 ML Models Used
| Model | Accuracy |
|-------|----------|
| K-Nearest Neighbors (KNN) | 93.33% |
| Decision Tree | 93.33% |
| **Support Vector Machine (SVM)** | **96.67%** ✅ Best |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/iris-classification.git
cd iris-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python iris_classification.py
```

---

## 📊 Dataset
- **Source**: [Kaggle - Iris CSV](https://www.kaggle.com/datasets/saurabh00007/iriscsv) / scikit-learn built-in
- **Samples**: 150 (50 per species)
- **Features**: 4 numerical features
- **No missing values**

---

## 📈 Results
- **Best Model**: Support Vector Machine (SVM) with **96.67% accuracy**
- Setosa is perfectly separable; versicolor and virginica have slight overlap

---

## 🛠️ Libraries Used
- `scikit-learn` — ML models, evaluation metrics
- `pandas` & `numpy` — Data manipulation
- `matplotlib` & `seaborn` — Visualization

---

## 📚 Concepts Learned
- Supervised Classification
- Train/Test Split & Stratification
- Feature Scaling (StandardScaler)
- Model Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Comparing multiple ML algorithms
