# =========================
# STREAMLIT PREMIUM UI
# CNN GENDER CLASSIFIER
# =========================

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

# =========================
# CONFIGURACIÓN
# =========================

st.set_page_config(
    page_title="Clasificador Facial IA",
    layout="wide",
    page_icon="🧠"
)

# =========================
# CSS PERSONALIZADO
# =========================

st.markdown("""
<style>

.stApp{
    background-color:#050505;
    color:white;
}

/* ocultar streamlit */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* HERO */

.hero-title{
    font-size:58px;
    font-weight:800;
    color:white;
    line-height:1.1;
    margin-top:80px;
}

.hero-sub{
    font-size:20px;
    color:#d1d5db;
    line-height:1.8;
    margin-top:25px;
}

/* UPLOAD */

.center-section{
    text-align:center;
    margin-top:140px;
    margin-bottom:100px;
}

.upload-title{
    font-size:42px;
    font-weight:bold;
    color:white;
}

.upload-sub{
    color:#9ca3af;
    font-size:18px;
    margin-top:10px;
}

/* RESULTADOS */

.result-card{

    background:linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #1e3a8a
    );

    padding:40px;

    border-radius:30px;

    text-align:center;

    color:white;

    box-shadow:0px 10px 30px rgba(0,0,0,0.4);
}

.result-title{
    font-size:22px;
    opacity:0.9;
}

.result-value{
    font-size:50px;
    font-weight:bold;
}

/* TÍTULOS */

.section-title{
    font-size:40px;
    font-weight:bold;
    color:white;
    margin-top:80px;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(
    "modelo_original.h5"
)

# =========================
# HERO SECTION
# =========================

col1, col2 = st.columns([1.1,1])

with col1:

    st.markdown(
        """
        <div class='hero-title'>
        Clasificador Inteligente de Rostros Masculinos y Femeninos
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='hero-sub'>

        Esta aplicación utiliza Redes Neuronales Convolucionales (CNN) y técnicas de inteligencia artificial para analizar imágenes faciales y estimar probabilidades de clasificación entre rostros masculinos y femeninos.

        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.image(
        "hero_image.png",
        width="600"
    )

# =========================
# ESPACIO
# =========================

st.write("")
st.write("")
st.write("")
st.write("")

# =========================
# SUBIR IMAGEN
# =========================

st.markdown(
    """
    <div class='center-section'>

    <div class='upload-title'>
    Sube una imagen de un rostro
    </div>

    <div class='upload-sub'>
    La inteligencia artificial analizará la imagen y mostrará la clasificación junto con los mapas de interpretabilidad.
    </div>

    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Subir imagen",
    type=["jpg","jpeg","png"]
)

# =========================
# SI HAY IMAGEN
# =========================

if uploaded_file is not None:

    # =========================
    # CARGAR IMAGEN
    # =========================

    img = Image.open(uploaded_file)

    img = img.convert("RGB")

    # =========================
    # PREPROCESAMIENTO
    # =========================

    img_resized = img.resize((128,128))

    img_array = np.array(img_resized)

    img_array = img_array / 255.0

    img_tensor = np.expand_dims(
        img_array,
        axis=0
    )

    # =========================
    # PREDICCIÓN
    # =========================

    with st.spinner("🧠 Analizando rostro con IA..."):

        prediction = model.predict(img_tensor)

        probability_male = float(prediction[0][0])

        probability_female = 1 - probability_male

        if probability_male > 0.5:

            predicted_class = "Masculino"

            confidence = probability_male

        else:

            predicted_class = "Femenino"

            confidence = probability_female

    # =========================
    # RESULTADO
    # =========================

    st.markdown(
        "<div class='section-title'>Resultado del Modelo</div>",
        unsafe_allow_html=True
    )

    col_img, col_pred = st.columns([1,1])

    # =========================
    # IMAGEN
    # =========================

    with col_img:

        st.image(
            img,
            width=400
        )

    # =========================
    # TARJETA RESULTADO
    # =========================

    with col_pred:

        st.markdown(
            f"""
            <div class='result-card'>

            <div class='result-title'>
            Clasificación
            </div>

            <div class='result-value'>
            {predicted_class}
            </div>

            <br>

            <div class='result-title'>
            Confianza del modelo
            </div>

            <div class='result-value'>
            {confidence*100:.2f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.progress(
            float(confidence)
        )

        st.write("")

        st.markdown(
            f"""
            ### 📊 Probabilidades por clase

            - Masculino: **{probability_male*100:.2f}%**
            - Femenino: **{probability_female*100:.2f}%**
            """
        )

    # =========================
    # SALIENCY MAP
    # =========================

    img_tensor_tf = tf.convert_to_tensor(
        img_tensor,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        tape.watch(img_tensor_tf)

        predictions = model(img_tensor_tf)

        loss = predictions[:,0]

    grads = tape.gradient(
        loss,
        img_tensor_tf
    )

    saliency = tf.reduce_max(
        tf.abs(grads),
        axis=-1
    )[0]

    saliency = saliency.numpy()

    # =========================
    # GRAD CAM
    # =========================

    activation_model = tf.keras.models.Model(
        inputs=model.layers[0].input,
        outputs=model.layers[4].output
    )

    feature_maps = activation_model.predict(
        img_tensor
    )

    heatmap = np.mean(
        feature_maps[0],
        axis=-1
    )

    heatmap = np.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / np.max(
        heatmap
    )

    heatmap = cv2.resize(
        heatmap,
        (128,128)
    )

    # =========================
    # INTERPRETABILIDAD
    # =========================

    st.markdown(
        "<div class='section-title'>Interpretabilidad del Modelo</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='
        color:#d1d5db;
        font-size:18px;
        line-height:1.8;
        margin-bottom:30px;
        '>

        En esta sección se visualizan mapas de activación generados a partir de la imagen original.

        Estas visualizaciones permiten identificar qué regiones faciales fueron más relevantes para que la red neuronal realizara la clasificación entre rostro masculino y femenino.

        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # COLUMNAS XAI
    # =========================

    col_xai1, col_xai2 = st.columns(2)

    # =========================
    # SALIENCY
    # =========================

    with col_xai1:

        fig1, ax1 = plt.subplots(
            figsize=(6,6)
        )

        fig1.patch.set_facecolor("#050505")

        ax1.imshow(img_array)

        ax1.imshow(
            saliency,
            cmap='hot',
            alpha=0.5
        )

        ax1.axis("off")

        ax1.set_title(
            "Saliency Map",
            fontsize=18,
            color="white"
        )

        st.pyplot(fig1)

    # =========================
    # GRAD CAM
    # =========================

    with col_xai2:

        fig2, ax2 = plt.subplots(
            figsize=(6,6)
        )

        fig2.patch.set_facecolor("#050505")

        ax2.imshow(img_array)

        ax2.imshow(
            heatmap,
            cmap='jet',
            alpha=0.5
        )

        ax2.axis("off")

        ax2.set_title(
            "Grad-CAM",
            fontsize=18,
            color="white"
        )

        st.pyplot(fig2)

# =========================
# FOOTER
# =========================

st.write("")
st.write("")
st.write("")

st.markdown("---")

st.markdown(
    """
    <center style='color:#9ca3af;'>

    Desarrollado con TensorFlow, CNNs e Inteligencia Artificial Explicable

    </center>
    """,
    unsafe_allow_html=True
)