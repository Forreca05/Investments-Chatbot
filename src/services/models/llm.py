import openai
import os
from dotenv import load_dotenv

# Carregando variáveis de ambiente do arquivo .env
load_dotenv(override=True)

class LLM:
    """Handles interactions with the OpenAI LLM (Large Language Model).

    Attributes:
        client (OpenAI): The OpenAI client instance.
        model_name (str): The name of the OpenAI LLM model to use.

    Methods:
        get_response(history, context, user_input): Generates a response from the LLM based on the conversation history, context, and user input.
    """
    
    def __init__(self):
        """Initializes the LLM class with OpenAI client and model information."""
        # Configuração do cliente OpenAI
        openai.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4")

    def get_response(self, history, context, user_input):
        """Generates a response from the LLM with streaming support.

        Args:
            history (list): A list of previous messages in the conversation history.
            context (str): Relevant information from the knowledge base to provide context to the LLM.
            user_input (str): The user's current input.

        Returns:
            str: The LLM's generated response.
        """
        messages = [
            {"role": "system", "content": context},
        ] + history + [
            {"role": "user", "content": user_input},
        ]

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )

        full_response = ""
        for event in response:
            if event.get("choices"):
                delta = event["choices"][0].get("delta", {})
                if "content" in delta:
                    full_response += delta["content"]
                    print(delta["content"], end="", flush=True)  # Print response in real-time
        
        return full_response
