#Iris flower classification.


#  STEP 1: Import Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("   IRIS FLOWER CLASSIFICATION - ML PROJECT")
print("=" * 60)


# STEP 2: Load and Explore the Dataset

print("\n[STEP 2] Loading Dataset...")

# Load iris dataset from sklearn (same as Kaggle CSV)
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDataset Info:")
print(df.info())
print(f"\nBasic Statistics:\n{df.describe()}")
print(f"\nSpecies Count:\n{df['species'].value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")


# STEP 3: Data Visualization

print("\n[STEP 3] Creating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Iris Dataset - Exploratory Data Analysis", fontsize=16, fontweight='bold')

colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}

# Plot 1: Sepal Length vs Sepal Width
ax1 = axes[0, 0]
for species, color in colors.items():
    subset = df[df['species'] == species]
    ax1.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                label=species, color=color, alpha=0.7, s=60)
ax1.set_xlabel('Sepal Length (cm)')
ax1.set_ylabel('Sepal Width (cm)')
ax1.set_title('Sepal: Length vs Width')

ax1.legend()

# Plot 2: Petal Length vs Petal Width
ax2 = axes[0, 1]
for species, color in colors.items():
    subset = df[df['species'] == species]
    ax2.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                label=species, color=color, alpha=0.7, s=60)
ax2.set_xlabel('Petal Length (cm)')
ax2.set_ylabel('Petal Width (cm)')
ax2.set_title('Petal: Length vs Width')
ax2.legend()


# Plot 3: Species Distribution
ax3 = axes[0, 2]
species_counts = df['species'].value_counts()
bars = ax3.bar(species_counts.index, species_counts.values,
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', alpha=0.85)
ax3.set_title('Species Distribution')
ax3.set_ylabel('Count')
for bar, val in zip(bars, species_counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha='center', fontweight='bold')


# Plot 4: Feature Boxplots
ax4 = axes[1, 0]
df_melted = df.melt(id_vars='species', var_name='Feature', value_name='Value')
species_list = ['setosa', 'versicolor', 'virginica']
color_list = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bp = ax4.boxplot(
    [df[df['species'] == s]['petal length (cm)'].values for s in species_list],
    labels=species_list, patch_artist=True
)
for patch, color in zip(bp['boxes'], color_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_title('Petal Length by Species')
ax4.set_ylabel('Petal Length (cm)')


# Plot 5: Correlation Heatmap
ax5 = axes[1, 1]
corr = df.drop('species', axis=1).corr()
im = ax5.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax5.set_xticks(range(len(corr.columns)))
ax5.set_yticks(range(len(corr.columns)))
ax5.set_xticklabels(['S.Len', 'S.Wid', 'P.Len', 'P.Wid'], rotation=45)
ax5.set_yticklabels(['S.Len', 'S.Wid', 'P.Len', 'P.Wid'])
for i in range(len(corr)):
    for j in range(len(corr)):
        ax5.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax5)
ax5.set_title('Feature Correlation Heatmap')


# Plot 6: Sepal Length Distribution
ax6 = axes[1, 2]
for species, color in colors.items():
    subset = df[df['species'] == species]['sepal length (cm)']
    ax6.hist(subset, bins=12, alpha=0.6, label=species, color=color, edgecolor='black')
ax6.set_xlabel('Sepal Length (cm)')
ax6.set_ylabel('Frequency')
ax6.set_title('Sepal Length Distribution')
ax6.legend()

plt.tight_layout()
plt.savefig('iris_eda.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: iris_eda.png")


# STEP 4: Prepare Data for ML

print("\n[STEP 4] Preparing Data...")

X = iris.data          # Features: sepal length, sepal width, petal length, petal width
y = iris.target        # Labels: 0=setosa, 1=versicolor, 2=virginica

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for KNN and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")


# STEP 5: Train Multiple ML Models

print("\n[STEP 5] Training Models...")

models = {
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"            : DecisionTreeClassifier(random_state=42),
    "Support Vector Machine"   : SVC(kernel='rbf', random_state=42),
}

results = {}
for name, model in models.items():
    if name in ["K-Nearest Neighbors (KNN)", "Support Vector Machine"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "predictions": y_pred, "accuracy": acc}
    print(f"   {name:<35} Accuracy: {acc*100:.2f}%")


# STEP 6: Detailed Evaluation - Best Model

best_name = max(results, key=lambda k: results[k]["accuracy"])
best = results[best_name]
print(f"\n[STEP 6] Best Model: {best_name}  ({best['accuracy']*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, best["predictions"], target_names=iris.target_names))


# STEP 7: Confusion Matrices for All Models

print("\n[STEP 7] Generating Confusion Matrices...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["predictions"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}\nAccuracy: {res['accuracy']*100:.2f}%", fontsize=10)

plt.tight_layout()
plt.savefig('iris_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: iris_confusion_matrices.png")


# STEP 8: Model Comparison Bar Chart

fig, ax = plt.subplots(figsize=(8, 5))
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]
bar_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(names, accs, color=bar_colors, edgecolor='black', alpha=0.85)
ax.set_ylim(90, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight='bold')
ax.set_xticklabels(names, rotation=10, ha='right')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{acc:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('iris_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: iris_model_comparison.png")


# STEP 9: Make a Sample Prediction

print("\n[STEP 9] Sample Prediction Demo:")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # Known setosa
sample_scaled = scaler.transform(sample)

for name, res in results.items():
    if name in ["K-Nearest Neighbors (KNN)", "Support Vector Machine"]:
        pred = res["model"].predict(sample_scaled)
    else:
        pred = res["model"].predict(sample)
    print(f"   {name:<35} → Predicted: {iris.target_names[pred[0]]}")

print("\n" + "=" * 60)
print("   ALL STEPS COMPLETE! Files saved:")
print("   • iris_eda.png")
print("   • iris_confusion_matrices.png")
print("   • iris_model_comparison.png")
print("=" * 60)
