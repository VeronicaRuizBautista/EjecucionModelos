import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Definir la clase para manejar múltiples modelos
class MultiModelHandler:
    def __init__(self):
        # Inicializar pipelines de diferentes tareas
        self.models = {
            "Traducir Texto": pipeline("translation_en_to_fr"),  # Traducción inglés a francés
            "Crear Imagen": StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
            ).to("cuda" if torch.cuda.is_available() else "cpu"),  # Generación de imágenes
        }

    def predict(self, task, *args, **kwargs):
        model_pipeline = self.models[task]
        return model_pipeline(*args, **kwargs)

# Inicializar el manejador de modelos
handler = MultiModelHandler()

# Interfaz con Streamlit
st.title("📚 Traducir Texto y Crear Imágenes 🖼️")

# Selección de tarea
task_option = st.selectbox(
    "Selecciona la tarea a realizar:", 
    ["Traducir Texto", "Crear Imagen", "Usar Ambos Modelos"]
)

if task_option == "Traducir Texto":
    input_text = st.text_area("Introduce el texto en inglés para traducir al francés:")
    if st.button("Traducir"):
        if input_text:
            with st.spinner("Traduciendo texto..."):
                response = handler.predict("Traducir Texto", input_text)
                st.success("Texto traducido:")
                st.write(response[0]['translation_text'])
        else:
            st.error("Por favor, introduce un texto para traducir.")

elif task_option == "Crear Imagen":
    prompt = st.text_input("Introduce una descripción para generar una imagen:")
    if st.button("Generar Imagen"):
        if prompt:
            with st.spinner("Generando imagen..."):
                image = handler.predict("Crear Imagen", prompt=prompt).images[0]
                st.success("Imagen generada:")
                st.image(image)
        else:
            st.error("Por favor, introduce una descripción para generar una imagen.")

elif task_option == "Usar Ambos Modelos":
    input_text = st.text_area("Introduce el texto en inglés para traducir y generar una imagen:")
    if st.button("Traducir y Generar Imagen"):
        if input_text:
            with st.spinner("Traduciendo texto..."):
                translation = handler.predict("Traducir Texto", input_text)
                translated_text = translation[0]['translation_text']
                st.success("Texto traducido:")
                st.write(translated_text)

            with st.spinner("Generando imagen basada en la traducción..."):
                image = handler.predict("Crear Imagen", prompt=translated_text).images[0]
                st.success("Imagen generada:")
                st.image(image)
        else:
            st.error("Por favor, introduce un texto para procesar.")
