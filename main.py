import os
import openai
from dotenv import load_dotenv
from src.services.models.embeddings import Embeddings
from src.services.vectorial_db.faiss_index import FAISSIndex
from src.services.models.llm import LLM

# Carregando variáveis de ambiente do arquivo .env
load_dotenv(override=True)

def rag_chatbot(llm: LLM, input_text: str, history: list, index: FAISSIndex):
    """Retrieves relevant information from the FAISS index, generates a response using the LLM, and manages the conversation history.

    Args:
        llm (LLM): An instance of the LLM class for generating responses.
        input_text (str): The user's input text.
        history (list): A list of previous messages in the conversation history.
        index (FAISSIndex): An instance of the FAISSIndex class for retrieving relevant information.

    Returns:
        tuple: A tuple containing the AI's response and the updated conversation history.
    """
    
    # Step 1: Convert input_text into embeddings
    input_embedding = index.embeddings(input_text)
    
    # Step 2: Retrieve the most relevant documents from FAISS index
    retrieved_context = index.retrieve_chunks(input_text, num_chunks=3)

    # Step 3: Format context for LLM input
    context_text = "\n".join(retrieved_context)
    
    # Step 4: Pass retrieved context and history to the LLM
    response_list = llm.get_response(history, context_text, input_text)
    ai_response = response_list[0]["content"]  # Aqui pegamos o conteúdo da primeira mensagem
    
    # Step 5: Update history with user input and AI response
    history.append({"role": "user", "content": input_text})
    history.append({"role": "assistant", "content": ai_response})
    
    return ai_response, history


def main():
    """Main function to run the chatbot."""
    
    # Carregar embeddings e o índice FAISS
    embeddings = Embeddings()
    index = FAISSIndex(embeddings=embeddings.get_embeddings)
    
    try:
        index.load_index()
    except FileNotFoundError:
        raise ValueError("Index not found. You must ingest documents first.")
    
    # Inicializar LLM e histórico de conversa
    llm = LLM()
    history = []
    print("\n# INITIALIZED CHATBOT #")
    
    while True:
        user_input = str(input("You:  "))
        if user_input.lower() == "exit":
            break
        response, history = rag_chatbot(llm, user_input, history, index)
        
        print("AI:", response.strip())


if __name__ == "__main__":
    main()
