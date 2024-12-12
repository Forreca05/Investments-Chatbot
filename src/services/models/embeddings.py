import openai
import os

class Embeddings:
    """Handles interactions with the OpenAI Embeddings API.

    Attributes:
        client (OpenAI): The OpenAI client instance.
        model (str): The name of the OpenAI embedding model to use.

    Methods:
        get_embeddings(text): Generates embeddings for the given text using the OpenAI Embeddings API.
    """
    def __init__(self):
        
        """Initializes the Embeddings class with  OpenAI client and model information."""
        openai.api_key = os.getenv("EMBEDDINGS_API_KEY")

        self.model = os.getenv("EMBEDDINGS_MODEL_NAME")


    def get_embeddings(self, text: str) -> list[float]:
        """Generates embeddings for the given text.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            list: A list of floats representing the text embedding.
        """
        completion = openai.embeddings.create(
            input=text,
            model=self.model
        )
        
        return completion.data[0].embedding
