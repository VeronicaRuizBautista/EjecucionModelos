import streamlit as st
from transformers import pipeline

class MultiModelHandler:
    def __init__(self):
        self.translation_model = pipeline("translation", model="t5-small")
        self.summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")

    def translate_text(self, text):
        return self.translation_model(text)[0]['translation_text']

    def summarize_text(self, text):
        return self.summarization_model(text)[0]['summary_text']


handler = MultiModelHandler()

st.title("Traducción y Resumen de Texto")

input_text = st.text_area("Ingresa tu texto:", height=200)

if st.button("Generar Traducción y Resumen"):
    if input_text:
        # Traducir y resumir el texto
        translated_text = handler.translate_text(input_text)
        summarized_text = handler.summarize_text(input_text)

        # Mostrar ambos resultados
        st.subheader("Texto Traducido:")
        st.write(translated_text)

        st.subheader("Texto Resumido:")
        st.write(summarized_text)
    else:
        st.warning("Por favor, ingresa un texto para procesar.")
