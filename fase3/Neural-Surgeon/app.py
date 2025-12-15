import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import json
import tempfile
import zipfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

CLASS_LABELS = ["bisturi", "curva", "hemostatica", "pinca", "reta"]
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="NeuroNet PSO - Classificador de Instrumentos Cirúrgicos",
    page_icon="🔬",
    layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin-bottom: 1rem;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready {
        background-color: #28a745;
    }
    .status-waiting {
        background-color: #ffc107;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .augmentation-step {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.25rem;
        text-align: center;
    }
</style>
""",
            unsafe_allow_html=True)

st.markdown('<p class="main-header">🔬 NeuroNet PSO</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Classificador de Instrumentos Cirúrgicos com Transfer Learning</p>',
    unsafe_allow_html=True)


def preprocess_image(image, return_steps=False):
    steps = {}

    # Passo 1: Guardar original
    original = image.copy()
    steps['original'] = original

    # Passo 2: Garantir RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    steps['rgb_converted'] = image.copy()

    # Passo 3: REDIMENSIONAMENTO INTELIGENTE (Smart Resize)
    # Isto impede que a imagem fique "esborrachada"
    target_w, target_h = IMAGE_SIZE

    # Calcula a proporção para caber dentro de 224x224 sem distorcer
    ratio = min(target_w / image.width, target_h / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))

    # Redimensiona a imagem
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Cria um fundo preto quadrado 224x224
    new_im = Image.new("RGB", (target_w, target_h), (0, 0, 0))

    # Cola a imagem redimensionada no centro
    paste_x = (target_w - new_size[0]) // 2
    paste_y = (target_h - new_size[1]) // 2
    new_im.paste(image, (paste_x, paste_y))

    # Atualiza a imagem para o próximo passo
    image = new_im
    steps['resized'] = image.copy()

    # Passo 4: Transformar em Array e Normalizar
    img_array = np.array(image)

    # IMPORTANTE: Se no treino dividiste por 255, mantém esta linha.
    # Se usaste 'preprocess_input' do EfficientNet, terias de mudar isto.
    # Assumindo /255 como estava no teu código original:
    img_array = img_array.astype('float32') / 255.0

    # Cria imagem para visualização na aba de debug
    normalized_display = Image.fromarray((img_array * 255).astype(np.uint8))
    steps['normalized'] = normalized_display

    # Adiciona a dimensão do batch (fica 1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    if return_steps:
        return img_array, steps
    return img_array

@st.cache_resource
def load_model_from_bytes(model_bytes, model_format='h5'):
    try:
        if model_format == 'h5':
            with open('temp_model.h5', 'wb') as f:
                f.write(model_bytes)
            model = tf.keras.models.load_model('temp_model.h5')
            os.remove('temp_model.h5')
            return model, None, 'h5'
        elif model_format == 'savedmodel':
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, 'model.zip')
                with open(zip_path, 'wb') as f:
                    f.write(model_bytes)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                model_dirs = [
                    d for d in os.listdir(tmpdir) if
                    os.path.isdir(os.path.join(tmpdir, d)) and d != '__MACOSX'
                ]
                if model_dirs:
                    model_path = os.path.join(tmpdir, model_dirs[0])
                else:
                    model_path = tmpdir
                model = tf.keras.models.load_model(model_path)
                return model, None, 'savedmodel'
        else:
            return None, "Formato não suportado", None
    except Exception as e:
        if os.path.exists('temp_model.h5'):
            os.remove('temp_model.h5')
        return None, str(e), None


def get_confidence_class(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


def classify_single_image(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)

    if predictions.shape[-1] == 1:
        predicted_class = int(predictions[0][0] > 0.5)
        probabilities = [1 - predictions[0][0], predictions[0][0]]
    else:
        predicted_class = np.argmax(predictions[0])
        probabilities = predictions[0]

    if predicted_class < len(CLASS_LABELS):
        predicted_label = CLASS_LABELS[predicted_class]
    else:
        predicted_label = f"Classe {predicted_class}"

    confidence = float(
        probabilities[predicted_class]) if predicted_class < len(
            probabilities) else float(max(probabilities))

    prob_dict = {}
    for i, label in enumerate(CLASS_LABELS):
        if i < len(probabilities):
            prob_dict[label] = float(probabilities[i]) * 100
        else:
            prob_dict[label] = 0.0

    return {
        'predicted_label': predicted_label,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': prob_dict
    }


def export_results_csv(results):
    rows = []
    for result in results:
        row = {
            'Imagem': result['filename'],
            'Classe Prevista': result['predicted_label'],
            'Confiança (%)': f"{result['confidence'] * 100:.2f}"
        }
        for label in CLASS_LABELS:
            row[f'Prob. {label} (%)'] = f"{result['probabilities'].get(label, 0):.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def export_results_json(results):
    export_data = []
    for result in results:
        export_data.append({
            'filename': result['filename'],
            'predicted_label': result['predicted_label'],
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
    return json.dumps(export_data, indent=2, ensure_ascii=False)


tabs = st.tabs(
    ["🔬 Classificação", "📊 Métricas do Modelo", "🖼️ Pré-processamento"])

with tabs[0]:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown("### 📁 Carregar Modelo")

        model_format = st.selectbox(
            "Formato do Modelo",
            options=['h5', 'savedmodel'],
            format_func=lambda x: {
                'h5': 'Keras H5 (.h5)',
                'savedmodel': 'SavedModel (.zip)'
            }.get(x, x),
            help=
            "Selecione o formato do modelo: H5 para arquivos .h5, SavedModel para diretórios compactados em .zip"
        )

        file_types = ['h5'] if model_format == 'h5' else ['zip']
        model_file = st.file_uploader(
            "Selecione o arquivo do modelo",
            type=file_types,
            key="model_uploader",
            help="Faça upload de um modelo Keras/TensorFlow")

        st.markdown("---")
        st.markdown("### 📊 Status do Sistema")

        if model_file is not None:
            model_bytes = model_file.read()
            model, error, loaded_format = load_model_from_bytes(
                model_bytes, model_format)
            if model is not None:
                st.success(f"✅ Modelo {loaded_format.upper()} carregado!")
                st.session_state['model'] = model
                st.session_state['model_loaded'] = True
                st.session_state['model_format'] = loaded_format

                try:
                    input_shape = model.input_shape
                    output_shape = model.output_shape
                    st.info(f"📐 Entrada: {input_shape}")
                    st.info(f"📤 Saída: {output_shape}")
                except:
                    pass
            else:
                st.error(f"❌ Erro: {error}")
                st.session_state['model_loaded'] = False
        else:
            st.info("⏳ Aguardando upload do modelo...")
            st.session_state['model_loaded'] = False

    with col2:
        st.markdown("### 🖼️ Entrada de Dados")

        batch_mode = st.checkbox("📦 Modo Lote (múltiplas imagens)",
                                 value=False)

        if batch_mode:
            image_files = st.file_uploader(
                "Arraste e solte suas imagens aqui",
                type=['jpg', 'jpeg', 'png'],
                key="batch_image_uploader",
                accept_multiple_files=True,
                help="Selecione múltiplas imagens para classificação em lote")

            if image_files:
                st.caption(f"📁 {len(image_files)} imagens selecionadas")
                cols = st.columns(min(len(image_files), 4))
                for idx, img_file in enumerate(image_files[:4]):
                    with cols[idx]:
                        img = Image.open(img_file)
                        st.image(img,
                                 caption=img_file.name[:15],
                                 width="stretch")
                if len(image_files) > 4:
                    st.caption(f"... e mais {len(image_files) - 4} imagens")
        else:
            image_files = st.file_uploader(
                "Arraste e solte sua imagem aqui",
                type=['jpg', 'jpeg', 'png'],
                key="single_image_uploader",
                help="Formatos suportados: JPG, PNG, JPEG")

            if image_files is not None:
                image = Image.open(image_files)
                st.image(image, caption="Imagem carregada", width="stretch")

        st.markdown("---")

        if not st.session_state.get('model_loaded', False):
            st.warning("⚠️ Por favor, carregue o modelo primeiro.")

        has_images = (batch_mode
                      and image_files) or (not batch_mode
                                           and image_files is not None)

        classify_button = st.button(
            "▶️ Executar Classificação",
            disabled=not st.session_state.get('model_loaded', False)
            or not has_images,
            width="stretch")

    with col3:
        st.markdown("### 📈 Resultados")

        if classify_button and st.session_state.get('model_loaded',
                                                    False) and has_images:
            model = st.session_state['model']
            all_results = []

            if batch_mode and image_files:
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, img_file in enumerate(image_files):
                    status_text.text(
                        f"Processando {idx + 1}/{len(image_files)}...")
                    progress_bar.progress((idx + 1) / len(image_files))

                    try:
                        img_file.seek(0)
                        image = Image.open(img_file)
                        result = classify_single_image(image, model)
                        result['filename'] = img_file.name
                        all_results.append(result)
                    except Exception as e:
                        all_results.append({
                            'filename': img_file.name,
                            'error': str(e)
                        })

                status_text.text("✅ Processamento completo!")
                st.session_state['batch_results'] = all_results

                st.markdown("#### 📊 Resumo do Lote")

                successful = [r for r in all_results if 'error' not in r]
                failed = [r for r in all_results if 'error' in r]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Sucesso", len(successful))
                with col_b:
                    st.metric("Erros", len(failed))

                if successful:
                    class_counts = {}
                    for r in successful:
                        label = r['predicted_label']
                        class_counts[label] = class_counts.get(label, 0) + 1

                    st.markdown("#### Distribuição por Classe")
                    for label, count in sorted(class_counts.items(),
                                               key=lambda x: -x[1]):
                        st.write(f"🔹 {label}: {count}")

                    avg_conf = sum(r['confidence']
                                   for r in successful) / len(successful)
                    st.metric("Confiança Média", f"{avg_conf * 100:.1f}%")

                st.markdown("---")
                st.markdown("#### 📥 Exportar Resultados")

                col_csv, col_json = st.columns(2)
                with col_csv:
                    csv_data = export_results_csv(successful)
                    st.download_button(
                        "📄 Download CSV",
                        data=csv_data,
                        file_name="resultados_classificacao.csv",
                        mime="text/csv",
                        width="stretch")
                with col_json:
                    json_data = export_results_json(successful)
                    st.download_button(
                        "📋 Download JSON",
                        data=json_data,
                        file_name="resultados_classificacao.json",
                        mime="application/json",
                        width="stretch")

            else:
                with st.spinner("Processando imagem..."):
                    try:
                        image_files.seek(0)
                        image = Image.open(image_files)
                        result = classify_single_image(image, model)
                        result['filename'] = image_files.name
                        all_results = [result]
                        st.session_state['batch_results'] = all_results

                        predicted_label = result['predicted_label']
                        confidence = result['confidence']

                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🎯 Previsão</h2>
                            <h1>{predicted_label}</h1>
                        </div>
                        """,
                                    unsafe_allow_html=True)

                        confidence_class = get_confidence_class(confidence)
                        st.markdown(
                            f'<p class="{confidence_class}">Confiança: {confidence * 100:.1f}%</p>',
                            unsafe_allow_html=True)

                        st.markdown("#### 📊 Probabilidades")

                        prob_values = [
                            result['probabilities'].get(label, 0)
                            for label in CLASS_LABELS
                        ]
                        chart_df = pd.DataFrame(
                            {'Probabilidade (%)': prob_values},
                            index=pd.Index(CLASS_LABELS))

                        st.bar_chart(chart_df)

                        st.markdown("#### 📋 Detalhes")
                        for label in CLASS_LABELS:
                            prob = result['probabilities'].get(label, 0)
                            emoji = "✅" if label == predicted_label else "🔹"
                            st.write(f"{emoji} {label}: {prob:.1f}%")

                        st.markdown("---")
                        st.markdown("#### 📥 Exportar")
                        col_csv, col_json = st.columns(2)
                        with col_csv:
                            csv_data = export_results_csv(all_results)
                            st.download_button("📄 CSV",
                                               data=csv_data,
                                               file_name="resultado.csv",
                                               mime="text/csv",
                                               width="stretch")
                        with col_json:
                            json_data = export_results_json(all_results)
                            st.download_button("📋 JSON",
                                               data=json_data,
                                               file_name="resultado.json",
                                               mime="application/json",
                                               width="stretch")

                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
        else:
            st.info("📊 Faça upload de uma imagem para ver os resultados.")

with tabs[1]:
    st.markdown("### 📊 Métricas do Modelo")

    if st.session_state.get('model_loaded', False):
        model = st.session_state['model']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            try:
                num_layers = len(model.layers)
                st.metric("Camadas", num_layers)
            except:
                st.metric("Camadas", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            try:
                num_params = model.count_params()
                if num_params > 1_000_000:
                    param_str = f"{num_params / 1_000_000:.2f}M"
                elif num_params > 1_000:
                    param_str = f"{num_params / 1_000:.1f}K"
                else:
                    param_str = str(num_params)
                st.metric("Parâmetros", param_str)
            except:
                st.metric("Parâmetros", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            try:
                trainable = sum([
                    tf.keras.backend.count_params(w)
                    for w in model.trainable_weights
                ])
                if trainable > 1_000_000:
                    train_str = f"{trainable / 1_000_000:.2f}M"
                elif trainable > 1_000:
                    train_str = f"{trainable / 1_000:.1f}K"
                else:
                    train_str = str(trainable)
                st.metric("Parâmetros Treináveis", train_str)
            except:
                st.metric("Parâmetros Treináveis", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏗️ Arquitetura do Modelo")

        try:
            layer_info = []
            for i, layer in enumerate(model.layers):
                try:
                    output_shape = str(layer.output_shape) if hasattr(
                        layer, 'output_shape') else "N/A"
                    params = layer.count_params() if hasattr(
                        layer, 'count_params') else 0
                    layer_info.append({
                        '#': i + 1,
                        'Nome': layer.name,
                        'Tipo': layer.__class__.__name__,
                        'Saída': output_shape,
                        'Parâmetros': params
                    })
                except:
                    layer_info.append({
                        '#':
                        i + 1,
                        'Nome':
                        layer.name if hasattr(layer, 'name') else f"layer_{i}",
                        'Tipo':
                        layer.__class__.__name__,
                        'Saída':
                        "N/A",
                        'Parâmetros':
                        0
                    })

            if layer_info:
                df = pd.DataFrame(layer_info)
                st.dataframe(df, width="stretch", hide_index=True)
        except Exception as e:
            st.warning(
                f"Não foi possível extrair informações das camadas: {str(e)}")

        st.markdown("---")
        st.markdown("### 📈 Estatísticas de Classificação")

        if 'batch_results' in st.session_state and st.session_state[
                'batch_results']:
            results = st.session_state['batch_results']
            successful = [r for r in results if 'error' not in r]

            if successful:
                col1, col2 = st.columns(2)

                with col1:
                    class_counts = {}
                    for r in successful:
                        label = r['predicted_label']
                        class_counts[label] = class_counts.get(label, 0) + 1

                    st.markdown("#### Distribuição de Classes")
                    count_df = pd.DataFrame(
                        {'Contagem': list(class_counts.values())},
                        index=pd.Index(list(class_counts.keys())))
                    st.bar_chart(count_df)

                with col2:
                    confidences = [r['confidence'] * 100 for r in successful]
                    st.markdown("#### Estatísticas de Confiança")
                    st.metric("Média", f"{np.mean(confidences):.1f}%")
                    st.metric("Mínimo", f"{np.min(confidences):.1f}%")
                    st.metric("Máximo", f"{np.max(confidences):.1f}%")
                    st.metric("Desvio Padrão", f"{np.std(confidences):.1f}%")
            else:
                st.info("Execute classificações para ver estatísticas.")
        else:
            st.info("Execute classificações para ver estatísticas.")
    else:
        st.info("⏳ Carregue um modelo para ver as métricas.")

with tabs[2]:
    st.markdown("### 🖼️ Visualização do Pré-processamento")
    st.markdown(
        "Veja como suas imagens são transformadas antes da classificação.")

    preview_image = st.file_uploader(
        "Selecione uma imagem para visualizar o pré-processamento",
        type=['jpg', 'jpeg', 'png'],
        key="preview_uploader")

    if preview_image is not None:
        image = Image.open(preview_image)
        _, steps = preprocess_image(image, return_steps=True)

        st.markdown("#### Etapas do Pré-processamento")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="augmentation-step">',
                        unsafe_allow_html=True)
            st.markdown("**1. Original**")
            st.image(steps['original'], width="stretch")
            orig_size = steps['original'].size
            st.caption(f"Tamanho: {orig_size[0]}x{orig_size[1]}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="augmentation-step">',
                        unsafe_allow_html=True)
            st.markdown("**2. RGB Convertido**")
            st.image(steps['rgb_converted'], width="stretch")
            st.caption(f"Modo: RGB")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="augmentation-step">',
                        unsafe_allow_html=True)
            st.markdown("**3. Redimensionado**")
            st.image(steps['resized'], width="stretch")
            st.caption(f"Tamanho: 224x224")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="augmentation-step">',
                        unsafe_allow_html=True)
            st.markdown("**4. Normalizado**")
            st.image(steps['normalized'], width="stretch")
            st.caption("Valores: 0-1")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Informações da Imagem")

        col_info1, col_info2, col_info3 = st.columns(3)

        with col_info1:
            st.markdown("**Dimensões Originais**")
            st.write(f"Largura: {steps['original'].size[0]}px")
            st.write(f"Altura: {steps['original'].size[1]}px")

        with col_info2:
            st.markdown("**Dimensões Finais**")
            st.write(f"Largura: 224px")
            st.write(f"Altura: 224px")
            st.write(f"Canais: 3 (RGB)")

        with col_info3:
            st.markdown("**Formato do Tensor**")
            st.write(f"Shape: (1, 224, 224, 3)")
            st.write(f"Tipo: float32")
            st.write(f"Range: [0.0, 1.0]")

        st.markdown("---")
        st.markdown("#### 🔍 Histograma de Cores")

        img_array = np.array(steps['resized'])

        hist_data = pd.DataFrame({
            'Vermelho':
            np.histogram(img_array[:, :, 0], bins=50, range=(0, 255))[0],
            'Verde':
            np.histogram(img_array[:, :, 1], bins=50, range=(0, 255))[0],
            'Azul':
            np.histogram(img_array[:, :, 2], bins=50, range=(0, 255))[0]
        })

        st.line_chart(hist_data)
    else:
        st.info(
            "📁 Faça upload de uma imagem para visualizar as etapas do pré-processamento."
        )

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>© 2025 NeuroNet Project. Meta III - Transfer Learning com PSO.</p>",
    unsafe_allow_html=True)
