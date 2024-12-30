from transformers import pipeline
import streamlit as st


qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


st.title("🤖 Preguntas y Respuestas con roberta-base-squad2 ")
question = st.text_input("Escribe tu pregunta:")
context = st.text_area("Proporciona el contexto:")

if st.button("Responder"):
    if question and context:
        response = qa_pipeline(question=question, context=context)
        st.success(f"Respuesta: {response['answer']}")
    else:
        st.error("Por favor, llena ambos campos.")
