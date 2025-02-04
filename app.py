import gradio as gr
import time
import os
import tempfile
from src.ingestion.loaders.loaderPDF import LoaderPDF
from src.ingestion.ingest_files import ingest_files_data_folder
from src.services.models.embeddings import Embeddings
from src.services.models.llm import LLM
from src.services.vectorial_db.faiss_index import FAISSIndex

# Inicializar instâncias para o LLM, embeddings e índice FAISS
llm = LLM()
embeddings = Embeddings()
index = FAISSIndex(embeddings=embeddings.get_embeddings)

# Variáveis globais para temperatura e tokens máximos
current_temperature = 1.0
current_max_tokens = 800

# Carregar o índice FAISS, ingeste dados se não existir
try:
    index.load_index()
except FileNotFoundError:
    ingest_files_data_folder(index)
    index.save_index()


def chatbot_wrapper(input_text, history):
    """
    Função wrapper para o chatbot, integrando com Gradio.

    Args:
        input_text (str): Texto de entrada do usuário.
        history (list): Histórico da conversa.

    Returns:
        tuple: Histórico atualizado e uma string de placeholder.
    """
    if history is None:
        history = []

    # Gerar a resposta do LLM com o histórico
    context = "Você é um assistente que ajuda com informações. Como posso te ajudar?"  # Altere conforme o necessário
    response = llm.get_response(history, context, input_text)
    
    # Adiciona a resposta do modelo ao histórico
    updated_history = history + [{"role": "assistant", "content": response[0]["content"]}]
    
    return updated_history, ""  # Retorna o histórico atualizado


def add_user_text(history, txt):
    """
    Adiciona o texto do usuário ao histórico da conversa.

    Args:
        history (list): Histórico da conversa.
        txt (str): Texto de entrada do usuário.

    Returns:
        tuple: Histórico atualizado e o texto de entrada do usuário.
    """
    if history is None:
        history = []
    history = history + [{"role": "user", "content": txt}]
    return history, txt


def update_temperature(temperature):
    """Atualiza a variável global da temperatura."""
    global current_temperature
    current_temperature = temperature

def update_max_tokens(max_tokens):
    """Atualiza a variável global dos tokens máximos."""
    global current_max_tokens
    current_max_tokens = max_tokens



def add_file(history, file_obj):
    """Processa o upload de arquivos e adiciona o conteúdo ao histórico."""
    # Verifica se o arquivo é um PDF
    if file_obj.name.endswith('.pdf'):
        # Cria um caminho temporário para salvar o arquivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_obj.read())
            temp_file_path = temp_file.name

        # Carrega o PDF e extrai o texto
        loader = LoaderPDF(temp_file_path)  # Passa o caminho temporário
        extracted_text = loader.extract_text()
        
        # Adiciona o conteúdo extraído ao histórico
        history.append({"role": "system", "content": extracted_text})
    
    return history


def process(history):
    """Função placeholder para processar a entrada."""
    return history


# Criar a interface Gradio
with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Roboto"), "Arial", "sans-serif"])) as demo:
    # Elemento de UI do Chatbot
    chatbot_ui = gr.Chatbot(
        [],
        type="messages",
        elem_id="Investment",
        bubble_full_width=True,
        height=950,
        avatar_images=("img/user.png", "img/gpt.png")  # Ajuste conforme as imagens
    )

    with gr.Row():
        # Caixa de texto para entrada
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Digite o texto e pressione Enter",
            container=True,
            interactive=True
        )

    with gr.Row():
        # Sliders para temperatura e tokens máximos
        temperature = gr.Slider(0.0, 2, value=1, label="Temperature")
        max_tokens = gr.Slider(1, 1000, value=800, label="Max Tokens")

        # Registrar os valores dos sliders (funções placeholder)
    t = temperature.release(update_temperature, inputs=[temperature])
    mt = max_tokens.release(update_max_tokens, inputs=[max_tokens])

    # Definir a cadeia de eventos: submeter texto -> adicionar ao histórico -> chamar chatbot_wrapper -> limpar a caixa de texto
    txt_msg = txt.submit(add_user_text, [chatbot_ui, txt], [chatbot_ui, txt]).then(
        chatbot_wrapper, [txt, chatbot_ui], [chatbot_ui, txt], queue=False
    ).then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

# Lançar a interface Gradio
demo.launch(share=True, auth=("joao", "ferreira"))
