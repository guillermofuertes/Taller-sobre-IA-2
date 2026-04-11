import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tiktoken

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(layout="wide")

# -------------------------
# ESTILO VISUAL
# -------------------------
st.markdown("""
<style>
.big-title {text-align:center; font-size:60px; font-weight:bold;}
.subtitle {text-align:center; font-size:28px;}
.result {color:red; font-size:40px; font-weight:bold;}
.box {padding:20px; border-radius:10px; background-color:#f5f5f5;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# MODELO
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------
# NAVEGACIÓN
# -------------------------
pagina = st.sidebar.radio(
    "Navegación",
    [
        "Inicio",
        "¿Qué ve la IA?",
        "Tokenizer",
        "Construir embeddings",
        "Embeddings reales",
        "Analogías",
        "Visualización"
    ]
)

# -------------------------
# INICIO
# -------------------------
if pagina == "Inicio":

    st.markdown("""
    <div class="big-title">
    Comenzar a entender<br>cómo funciona realmente la IA
    </div>
    <div class="subtitle">Guillermo Fuertes</div>
    """, unsafe_allow_html=True)

# -------------------------
# ¿QUÉ VE LA IA?
# -------------------------
elif pagina == "¿Qué ve la IA?":

    st.markdown("## ¿Qué ve realmente la IA?")

    placeholder = st.empty()

    placeholder.markdown("# gato")
    time.sleep(1.5)

    placeholder.markdown("# [0.21, -0.34, 0.88, ...]")
    time.sleep(1.5)

    st.info("Esto… es lo que la IA realmente procesa.")

# -------------------------
# TOKENIZER
# -------------------------
elif pagina == "Tokenizer":

    st.title("De texto a tokens")

    texto = st.text_area("Introduce un texto", "El gato duerme en el sofá")

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(texto)
    decoded = [encoding.decode([t]) for t in tokens]

    html = ""
    colores = ["#FFCCCC", "#CCFFCC", "#CCCCFF", "#FFFFCC"]

    for i, tok in enumerate(decoded):
        html += f"<span style='background-color:{colores[i%4]}; padding:8px; margin:4px; border-radius:6px;'>{tok} ({tokens[i]})</span>"

    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# CONSTRUIR EMBEDDINGS
# -------------------------
elif pagina == "Construir embeddings":

    st.title("Construyendo embeddings")

    dimensiones = [
        "Tiene cola", "Es animal", "Es comible",
        "Es fruta", "Es felino", "Transporte", "Es mascota"
    ]

    palabras = ["Gato", "Perro", "Tigre", "Manzana", "Autobus"]

    df = pd.DataFrame(0.0, index=dimensiones, columns=palabras)
    df_editado = st.data_editor(df)

    if st.button("Generar embeddings"):

        st.subheader("Vectores:")

        for col in df_editado.columns:
            vector = df_editado[col].values
            st.write(f"{col}: {vector}")

        st.subheader("Espacio semántico")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(df_editado.T)

        fig, ax = plt.subplots()
        placeholder = st.empty()

        for i, palabra in enumerate(df_editado.columns):
            x, y = reduced[i]
            ax.scatter(x, y)
            ax.text(x, y, palabra)
            placeholder.pyplot(fig)
            time.sleep(0.7)

# -------------------------
# EMBEDDINGS REALES
# -------------------------
elif pagina == "Embeddings reales":

    st.title("Embeddings reales")

    texto = st.text_input("Palabras", "gato, perro, coche")
    words = [w.strip() for w in texto.split(",")]

    embeddings = model.encode(words)

    st.subheader("Vectores:")

    for word, emb in zip(words, embeddings):
        vector = [float(f"{x:.3f}") for x in emb[:10]]
        st.write(f"{word}: {vector}")

    st.subheader("Similitudes")

    col1, col2 = st.columns(2)

    sim_data = []

    with col1:
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                label = f"{words[i]} vs {words[j]}"
                st.write(f"{label}: {sim:.2f}")
                sim_data.append((label, sim))

    with col2:
        if sim_data:
            labels = [x[0] for x in sim_data]
            values = [x[1] for x in sim_data]

            fig, ax = plt.subplots()
            ax.barh(labels, values)
            ax.set_xlim(0, 1)

            st.pyplot(fig)


# -------------------------
# VISUALIZACIÓN
# -------------------------
elif pagina == "Visualización":

    st.title("Espacio de embeddings real")

    st.markdown(
        """
        <iframe src="https://projector.tensorflow.org/" 
        width="100%" height="600"></iframe>
        """,
        unsafe_allow_html=True
    )
