# -*- coding: utf-8 -*-

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

import pyswarms as ps  

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

X = X.reshape(10000, 32, 32, 3)

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

def f_ackley(X):
    return np.array([ackley(xi) for xi in X])


print("\n**** A INICIAR OTIMIZAÇÃO PSO — ACKLEY 2D ****")
options_ack = {'c1': 1, 'c2': 1, 'w': 0.5}
bounds_ack2 = (np.array([-5, -5]), np.array([5, 5]))

optimizer_ack2 = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=2,
    options=options_ack,
    bounds=bounds_ack2
)

cost_ack2, pos_ack2 = optimizer_ack2.optimize(f_ackley, iters=30)

print("\n===== RESULTADO ACKLEY 2D =====")
print("Melhor posição encontrada:", pos_ack2)
print("Valor mínimo:", cost_ack2)

print("\n**** A INICIAR OTIMIZAÇÃO PSO — ACKLEY 3D ****")
bounds_ack3 = (np.array([-5, -5, -5]), np.array([5, 5, 5]))

optimizer_ack3 = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=3,
    options=options_ack,
    bounds=bounds_ack3
)

cost_ack3, pos_ack3 = optimizer_ack3.optimize(f_ackley, iters=30)

print("\n===== RESULTADO ACKLEY 3D =====")
print("Melhor posição encontrada:", pos_ack3)
print("Valor mínimo:", cost_ack3)


def evaluate_cnn(params):
    """
    AGORA SÓ 2 hiperparâmetros:
    params[0] = nº de filtros
    params[1] = nº de neurónios
    """
    n_filters = max(16, int(params[0]))
    n_neurons = max(32, int(params[1]))

    model = Sequential([
        Conv2D(n_filters, (3,3), activation='relu', input_shape=(32,32,3)),
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

def f(x):
    return np.array([evaluate_cnn(xi) for xi in x])

print("\n**** A INICIAR OTIMIZAÇÃO PSO — CNN (2 hiperparâmetros) ****")

options = {'c1': 1, 'c2': 1, 'w': 0.5}

min_bounds = [16, 32]
max_bounds = [64, 150]
bounds = (np.array(min_bounds), np.array(max_bounds))

optimizer = ps.single.GlobalBestPSO(
    n_particles=5,
    dimensions=2,     
    options=options,
    bounds=bounds
)

cost, pos = optimizer.optimize(f, iters=30) #50

best_filters = int(pos[0])
best_neurons = int(pos[1])
best_accuracy = 1 - cost

print("\n**** MELHOR CONFIGURAÇÃO ENCONTRADA ****")
print(f"Filtros: {best_filters}")
print(f"Neurónios: {best_neurons}")
print(f"Batch size (fixo): 25")
print(f"Learning rate (fixo): 0.01")
print(f"Accuracy estimada: {best_accuracy:.4f}")


# Treino final
final_model = Sequential([
    Conv2D(best_filters, (3,3), activation='relu', input_shape=(32,32,3)), 
    Flatten(), 
    Dense(best_neurons, activation='relu'),
    Dense(num_classes, activation='softmax')
])

final_model.compile(optimizer='adam', loss='categorical_crossentropy') 

final_model.fit(
    X_train, yTrain_binario,
    epochs=50,
    batch_size=25,     # fixo
    verbose=2
)

loss_final, acc_final = final_model.evaluate(X_test, yTest_binario, verbose=0)
print(f"\nAccuracy final no conjunto de teste: {acc_final:.4f}")


# Matriz de confusão
y_prob_final = final_model.predict(X_test)
y_pred_final = np.argmax(y_prob_final, axis=1)

print("\nRelatório final:")
print(classification_report(y_test, y_pred_final, target_names=le.classes_))

matriz_final = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(7,6))
sns.heatmap(matriz_final, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("CNN otimizada com PSO — 2 Hiperparâmetros")
plt.tight_layout()
plt.show()


# Métricas
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

acc_global = accuracy_score(y_test, y_pred_final)
recall_macro = recall_score(y_test, y_pred_final, average="macro")
f1_macro = f1_score(y_test, y_pred_final, average="macro")
auc_macro = roc_auc_score(yTest_binario, y_prob_final, multi_class="ovr")

specificities = []
for i in range(len(matriz_final)):
    tn = matriz_final.sum() - (matriz_final[i, :].sum() + matriz_final[:, i].sum() - matriz_final[i, i])
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
