import openai
import os


class LLM():
    """Handles interactions with the OpenAI LLM (Large Language Model).

    Attributes:
        client (OpenAI): The OpenAI client instance.
        model_name (str): The name of the OpenAI LLM model to use.

    Methods:
        get_response(history, context, user_input): Generates a response from the LLM based on the conversation history, context, and user input.
    """
    def __init__(self):
        """Initializes the LLM class with OpenAI client and model information."""
        # OpenAI client setup
        openai.api_key = os.getenv("LLM_API_KEY")
        
        self.model_name = os.getenv("LLM_MODEL_NAME")



    def get_response(self, history, context, user_input):
        """Generates a response from the LLM.

        Args:
            history (list): A list of previous messages in the conversation history.
            context (str): Relevant information from the knowledge base to provide context to the LLM.
            user_input (str): The user's current input.

        Returns:
            str: The LLM's generated response.
        """
        #XXX: NOT IMPLEMENTED. Use openai.chat.completions to create the chatbot response

        #TODO (EXTRA: stream LLM response)

        return "<AI RESPONSE PLACEHOLDER>"
