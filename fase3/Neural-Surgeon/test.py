# -- coding: utf-8 --
"""
Gera novas imagens com pequenas rotações, zoom leve,
brilho e ruído suave (pontinhos) para aumentar o dataset.
@author: inesr
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import os


import numpy as np

# Caminho da pasta original
input_dir = r"C:\Users\tiago\Documents\Tiago\Tiago\ISEC\3ªano\IC\TP\fase3\Reta\Reta"
output_dir = os.path.join(input_dir, "imgs_aumentadas")

# Cria a pasta de saída
os.makedirs(output_dir, exist_ok=True)

# --- Configuração do gerador (não corta tanto) ---
datagen = ImageDataGenerator(
    rotation_range=10,        # rotações suaves
    width_shift_range=0.05,   # deslocamento pequeno
    height_shift_range=0.05,
    zoom_range=0.05,          # zoom leve
    brightness_range=[0.9, 1.1],  # brilho mais natural
    shear_range=3,            # ligeira torção
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- Função para adicionar ruído ("pontinhos") ---
def add_noise(image_array, noise_intensity=0.02):
    """
    Adiciona pequenos pontos brancos e pretos aleatórios à imagem.
    """
    noisy = image_array.copy()
    # ruído gaussiano suave
    noise = np.random.normal(0, 255 * noise_intensity, noisy.shape)
    noisy = noisy + noise
    noisy = np.clip(noisy, 0, 255)  # garantir valores válidos de pixel
    return noisy.astype('uint8')

# Número de imagens aumentadas por original
num_augmented =15

# Processa cada imagem da pasta
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filepath = os.path.join(input_dir, filename)
        print(f"Aumentando {filename}...")

        # Carrega a imagem e converte para array
        img = load_img(filepath)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Gera novas imagens
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            augmented = batch[0]
            # Adiciona ruído leve
            augmented_noisy = add_noise(augmented, noise_intensity=0.02)

            # Guarda a imagem
            final_img = array_to_img(augmented_noisy)
            new_filename = f"{os.path.splitext(filename)[0]}aug{i+1}.jpg"
            final_img.save(os.path.join(output_dir, new_filename))

            i += 1
            if i >= num_augmented:
                break

print("\n✅ Aumento concluído! As novas imagens estão em:")
print(output_dir)