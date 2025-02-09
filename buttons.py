import gradio as gr
import time
import os
import tempfile
from src.services.models.embeddings import Embeddings
from src.services.models.llm import LLM
from src.services.vectorial_db.faiss_index import FAISSIndex

# Inicializar instâncias para o LLM, embeddings e índice FAISS
llm = LLM()
embeddings = Embeddings()
index = FAISSIndex(embeddings=embeddings.get_embeddings)


def update_temperature(temperature):
    """Atualiza a variável global da temperatura."""
    global current_temperature
    current_temperature = temperature


def update_max_tokens(max_tokens):
    """Atualiza a variável global dos tokens máximos."""
    global current_max_tokens
    current_max_tokens = max_tokens


def explain_better(history):
    """Pede ao LLM para gerar uma explicação mais detalhada da última resposta."""
    if not history or history[-1]["role"] != "assistant":
        return history  # Se não houver resposta do chatbot, não faz nada

    last_response = history[-1]["content"]
    prompt = f"Explique de forma mais detalhada a seguinte resposta: {last_response}"

    detailed_response = llm.get_response(history, "Explique melhor", prompt)

    history.append({"role": "assistant", "content": detailed_response[0]["content"]})
    return history


def summarize(history):
    """Pede ao LLM para resumir a última resposta."""
    if not history or history[-1]["role"] != "assistant":
        return history

    last_response = history[-1]["content"]
    prompt = f"Resuma a seguinte resposta de forma concisa: {last_response}"

    summary = llm.get_response(history, "Resumo", prompt)

    history.append({"role": "assistant", "content": summary[0]["content"]})
    return history


def generate_example(history):
    """Gera um exemplo baseado na última resposta."""
    if not history or history[-1]["role"] != "assistant":
        return history

    last_response = history[-1]["content"]
    prompt = f"Dê um exemplo prático baseado na seguinte resposta: {last_response}"

    example = llm.get_response(history, "Exemplo", prompt)

    history.append({"role": "assistant", "content": example[0]["content"]})
    return history
