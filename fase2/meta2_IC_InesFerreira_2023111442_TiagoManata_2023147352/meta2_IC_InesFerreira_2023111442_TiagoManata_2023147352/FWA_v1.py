# -*- coding: utf-8 -*-
"""
Fireworks Algorithm (FWA) + CNN + Ackley 2D/3D + Métricas completas
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
#from SwarmPackagePy import fwa
import fwa


def load_images(root, size=(32, 32)):
    X, y = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for file in folder.glob("*"):
            img = Image.open(file).convert("RGB").resize(size)
            X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(folder.name)
    return np.array(X), np.array(y)


DATA_DIR = Path(r"C:\Users\inesr\OneDrive\Ambiente de Trabalho\dataset")
np.random.seed(20)

X, y_text = load_images(DATA_DIR)
print(f"Total de imagens carregadas: {len(X)}")

le = LabelEncoder()
y = le.fit_transform(y_text)

X = X.reshape(len(X), 32, 32, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20, stratify=y)

num_classes = len(le.classes_)
yTrain_binario = to_categorical(y_train, num_classes)
yTest_binario = to_categorical(y_test, num_classes)

def ackley(x):
    x = np.array(x)
    n = len(x)
    part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    part2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return part1 + part2 + 20 + np.e


# -- ACKLEY 2D --
print("\n**** A INICIAR FWA — ACKLEY 2D ****")
fw_ack2 = fwa.fwa(
    n=30,   #alterar 50                    # número de fogos de artifício
    function=ackley,            # função a otimizar
    lb=[-5, -5],                      # limite inferior
    ub=[5, 5],                       # limite superior
    dimension=2,                # dimensão do problema
    iteration=30   #alterar 50              # iterações
)

pos_fwa2 = fw_ack2.get_Gbest()
cost_fwa2 = ackley(pos_fwa2)

print("\n===== RESULTADO FWA 2D =====")
print("Melhor posição:", pos_fwa2)
print("Valor mínimo:", cost_fwa2)

# -- ACKLEY 3D --
print("\n**** A INICIAR FWA — ACKLEY 3D ****")
fw_ack3 = fwa.fwa(
    n=30,     #alterar 50                    # número de fogos de artifício
    function=ackley,            # função a otimizar
    lb=[-5, -5, -5],                      # limite inferior
    ub=[5, 5, 5],                       # limite superior
    dimension=3,                # dimensão do problema
    iteration=30 #alterar 50                 # iterações
)

pos_fwa3 = fw_ack3.get_Gbest()
cost_fwa3 = ackley(pos_fwa3)

print("\n===== RESULTADO FWA 3D =====")
print("Melhor posição:", pos_fwa3)
print("Valor mínimo:", cost_fwa3)

def evaluate_cnn(params):
   
    n_filters = int(params[0])
    n_neurons = int(params[1])

    n_filters = int(np.clip(n_filters, 16, 64))
    n_neurons = int(np.clip(n_neurons, 32, 150))

    model = Sequential([
        Conv2D(n_filters, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        Flatten(),
        Dense(n_neurons, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy') 

    treino = model.fit(
        X_train, yTrain_binario,
        epochs=30, #50  
        batch_size=25,
        validation_split=0.2,
        verbose=1
    )

    acc = treino.history['val_accuracy'][-1]
    return 1 - acc

print("\n**** A INICIAR FWA — CNN ****")

min_bounds = np.array([16, 32])
max_bounds = np.array([64, 150])
dimension = 2

def f_firework_single(x):
    return evaluate_cnn(x)

# FWA
fw_cnn = fwa.fwa(
    n=5, #7
    function=f_firework_single,
    lb=min_bounds,
    ub=max_bounds,
    dimension=2,            
    iteration=10 #20
)

best_pos = fw_cnn.get_Gbest()
best_cost = f_firework_single(best_pos)
best_accuracy = 1 - best_cost

print("\n**** MELHOR CONFIGURAÇÃO FWA ****")
print("Filtros:", int(best_pos[0]))
print("Neurónios:", int(best_pos[1]))
print("Accuracy estimada:", best_accuracy)

print("\n**** A TREINAR MODELO FINAL ****")

best_filters = int(best_pos[0])
best_neurons = int(best_pos[1])

model_final = Sequential([
    Conv2D(best_filters, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    Flatten(),
    Dense(best_neurons, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model_final.compile(optimizer='adam', loss='categorical_crossentropy') 

model_final.fit(
    X_train, yTrain_binario,
    epochs=80, #alterar 100
    batch_size=25,
    validation_split=0.2,
    verbose=1
)

y_prob_final = model_final.predict(X_test)
y_pred_final = np.argmax(y_prob_final, axis=1)

print("\nRelatório final:")
print(classification_report(y_test, y_pred_final, target_names=le.classes_))

matriz_final = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(7, 6))
sns.heatmap(matriz_final, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("CNN otimizada com FWA — 4 Hiperparâmetros")
plt.tight_layout()
plt.show()

from sklearn.metrics import recall_score, f1_score, roc_auc_score

acc_global = accuracy_score(y_test, y_pred_final)
recall_macro = recall_score(y_test, y_pred_final, average="macro")
f1_macro = f1_score(y_test, y_pred_final, average="macro")
auc_macro = roc_auc_score(yTest_binario, y_prob_final, multi_class="ovr")

specificities = []
for i in range(len(matriz_final)):
    tn = matriz_final.sum() - (matriz_final[i, :].sum() +
                               matriz_final[:, i].sum() - matriz_final[i, i])
    fp = matriz_final[:, i].sum() - matriz_final[i, i]
    specificities.append(tn / (tn + fp))
spec_mean = np.mean(specificities)

print(f"\nAccuracy global: {acc_global:.3f}")
print(f"Recall macro: {recall_macro:.3f}")
print(f"F1-score macro: {f1_macro:.3f}")
print(f"Especificidade média: {spec_mean:.3f}")
print(f"AUC macro: {auc_macro:.3f}")

recall_classes = recall_score(y_test, y_pred_final, average=None)

print("\n**** Recall por classe ****")
for i, class_name in enumerate(le.classes_):
    print(f"Recall({class_name}) = {recall_classes[i]:.3f}")
