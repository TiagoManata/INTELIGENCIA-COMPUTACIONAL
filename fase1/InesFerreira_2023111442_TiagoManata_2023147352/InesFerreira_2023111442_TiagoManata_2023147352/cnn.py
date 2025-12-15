# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 2025
@author: inesr

"""

from pathlib import Path
import numpy as np
from PIL import Image

DATA_DIR = Path(r"C:\Users\tiago\Documents\Tiago\Tiago\ISEC\3ªano\IC\TP\fase1\InesFerreira_2023111442_TiagoManata_2023147352\InesFerreira_2023111442_TiagoManata_2023147352\dataset")
np.random.seed(20)

def load_images(root, size=(32, 32)):
    X, y = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for file in folder.glob("*"):
            img = Image.open(file).convert("RGB").resize(size)
            X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(folder.name)
    return np.array(X), np.array(y)

X, y_text = load_images(DATA_DIR)
print("Total de imagens:", len(X))

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(y_text)

X = X.reshape(10000, 32, 32, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y)

from tensorflow.keras.utils import to_categorical
yTrain_binario = to_categorical(y_train, 5)
yTest_binario = to_categorical(y_test, 5)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)), 
    #Conv2D(16, (3,3), activation='relu', input_shape=(32,32,3)),
    Flatten(), 
    Dense(100, activation='relu'),
    #Dense(200, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy') 

print("**** Treinar o modelo ****")
modelo_treino = model.fit(
    X_train, yTrain_binario,
    epochs=15,
    batch_size=16, 
    validation_split=0.1,
    verbose=2)

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

y_prob = model.predict(X_test)  
y_pred = np.argmax(y_prob, axis=1) 

acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
auc = roc_auc_score(yTest_binario, y_prob, multi_class="ovr")

matriz = confusion_matrix(y_test, y_pred)
# Calcular especificidade média (para várias classes)
specificity = []
for i in range(len(matriz)):
    tn_i = matriz.sum() - (matriz[i, :].sum() + matriz[:, i].sum() - matriz[i, i])
    fp_i = matriz[:, i].sum() - matriz[i, i]
    specificity.append(tn_i / (tn_i + fp_i))
specificity_mean = np.mean(specificity)

print(f"\nAccuracy: {acc:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"Especificidade: {specificity_mean:.3f}")
print(f"AUC: {auc:.3f}")

recallClasse = recall_score(y_test, y_pred, average=None)

print("\n**** Métricas por classe ****")
for i, class_name in enumerate(le.classes_):
    print(f"Recall({class_name})={recallClasse[i]:.3f} ")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Previstas')
plt.ylabel('Reais')
plt.title('Matriz de Confusão - CNN')
plt.tight_layout()
plt.show()
