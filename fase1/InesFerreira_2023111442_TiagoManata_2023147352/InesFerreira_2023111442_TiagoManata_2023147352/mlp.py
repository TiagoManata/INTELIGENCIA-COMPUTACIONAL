# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 20:20:20 2025

@author: inesr
"""

from pathlib import Path
DATA_DIR = Path(r"C:\Users\tiago\Documents\Tiago\Tiago\ISEC\3ªano\IC\TP\fase1\InesFerreira_2023111442_TiagoManata_2023147352\InesFerreira_2023111442_TiagoManata_2023147352\dataset")

import numpy as np
np.random.seed(20)

from PIL import Image
def load_images(root, size=(32, 32)):
    X, y = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for file in folder.glob("*"):
            try:
                img = Image.open(file).convert("L").resize(size)
                X.append(np.asarray(img, dtype=np.float32).ravel())
                y.append(folder.name)
            except:
                continue
    return np.array(X), np.array(y)

X, y_text = load_images(DATA_DIR) 
print("Total de imagens:", len(X))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_text)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y 
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
    solver="lbfgs",
    activation = "relu",
    hidden_layer_sizes=(100,),
    max_iter=100,
    validation_fraction=0.1,
    random_state=10
)
print("Treinar o modelo...")
model.fit(X_train_scaled, y_train)

from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

y_pred = model.predict(X_test_scaled) 
y_prob = model.predict_proba(X_test_scaled) 
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

matriz = confusion_matrix(y_test, y_pred)
specificity = []
for i in range(len(matriz)):
    tn_i = matriz.sum() - (matriz[i, :].sum() + matriz[:, i].sum() - matriz[i, i])
    fp_i = matriz[:, i].sum() - matriz[i, i]
    specificity.append(tn_i / (tn_i + fp_i))
specificity_mean = np.mean(specificity)

auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

print(f"\nAccuracy: {acc:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Especificidade: {specificity_mean:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC: {auc:.3f}")
    
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Previstas')
plt.ylabel('Reais')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.show()
