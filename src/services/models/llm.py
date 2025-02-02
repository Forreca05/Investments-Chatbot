import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Carregando variáveis de ambiente do arquivo .env
load_dotenv(override=True)

class LLM:
    """Handles interactions with the OpenAI LLM (Large Language Model)."""

    def __init__(self):
        """Inicializa o cliente OpenAI e define o modelo."""
        self.api_key = os.getenv("LLM_API_KEY")  # Garante que API Key está carregada
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")

    def get_response(self, history, context, user_input):
        """Gera uma resposta do LLM."""
        messages = [
            {"role": "system", "content": context},
        ] + history + [
            {"role": "user", "content": user_input},
        ]

        # Passando a API Key ao criar o cliente ✅
        client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model_name,  # Usa modelo configurado
            messages=messages,
            temperature=0.7
        )

        return [{"role": "assistant", "content": response.choices[0].message.content}]
