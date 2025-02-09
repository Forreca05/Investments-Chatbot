import gradio as gr
import time
import os
import tempfile
import buttons
import sqlite3
import sql
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
    sql.cleanup_db()
    index.load_index()
except FileNotFoundError:
    ingest_files_data_folder(index)
    index.save_index()

sql.initialize_db()

def chatbot_wrapper(input_text, history):
    """Gera uma resposta do LLM com o histórico."""
    if history is None:
        history = []

    context = "Você é um assistente que ajuda com informações. Como posso te ajudar?"
    response = llm.get_response(history, context, input_text)
    
    updated_history = history + [{"role": "assistant", "content": response[0]["content"]}]

    sql.save_to_db("user", input_text)
    sql.save_to_db("assistant", response[0]["content"])
    
    return updated_history, ""

# Função para carregar histórico salvo
def use_saved_history():
    """Retorna o histórico armazenado no banco de dados SQLite."""
    return sql.load_history_from_db()


def add_user_text(history, txt):
    """Adiciona o texto do usuário ao histórico da conversa."""
    if history is None:
        history = []
    history = history + [{"role": "user", "content": txt}]
    return history, txt

def add_file(history, file_obj):
    """Processa o upload de arquivos e adiciona o conteúdo ao histórico."""
    if file_obj.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_obj.read())
            temp_file_path = temp_file.name

        loader = LoaderPDF(temp_file_path)
        extracted_text = loader.extract_text()
        
        history.append({"role": "system", "content": extracted_text})
    
    return history


# Criar a interface Gradio
with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Roboto"), "Arial", "sans-serif"]), css="custom_css") as demo:
    
    chatbot_ui = gr.Chatbot(
        [],
        label="Investment Chatbot",
        show_copy_button=True,
        type="messages",
        elem_id="Investment",
        bubble_full_width=True,
        height=900,
        avatar_images=("img/user.png", "img/gpt.png")
    )

    # Linha com a caixa de texto e o botão ao lado, agora com o mesmo tamanho
    with gr.Row():
        txt = gr.Textbox(
            scale=4,  # Definindo o mesmo scale para o texto e botão
            show_label=False,
            placeholder="Digite o texto e pressione Enter",
            container=True,
            interactive=True
        )

        use_history_button = gr.Button(value="History", variant="secondary", scale=0.5, elem_id="use-history-button", size="lg", icon="img/investments.png")  # Alinhando o tamanho

    with gr.Row():
        temperature = gr.Slider(0.0, 2, value=1, label="Temperature")
        max_tokens = gr.Slider(1, 1000, value=800, label="Max Tokens")

    # Passando as funções de atualização corretamente para o release dos sliders
    t = temperature.release(buttons.update_temperature, inputs=[temperature], outputs=[])
    mt = max_tokens.release(buttons.update_max_tokens, inputs=[max_tokens], outputs=[])

    # Botões "Explicar Melhor", "Resumo" e "Exemplo"
    with gr.Row():
        explain_button = gr.Button(value="Explain Better", variant="primary", icon="https://upload.wikimedia.org/wikipedia/en/e/e1/Sporting_Clube_de_Portugal_%28Logo%29.svg", elem_id="explain-button")
        summarize_button = gr.Button(value="Resume", variant="huggingface", icon="https://upload.wikimedia.org/wikipedia/en/thumb/f/f1/FC_Porto.svg/640px-FC_Porto.svg.png")
        example_button = gr.Button(value="Example", variant="primary", icon="https://fpfimagehandler.fpf.pt/ScoreImageHandler.ashx?type=Organization&id=503")

    # Eventos do chatbot
    txt_msg = txt.submit(add_user_text, [chatbot_ui, txt], [chatbot_ui, txt]).then(
        chatbot_wrapper, [txt, chatbot_ui], [chatbot_ui, txt], queue=False
    ).then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    # Eventos dos botões
    explain_button.click(buttons.explain_better, [chatbot_ui], [chatbot_ui])
    summarize_button.click(buttons.summarize, [chatbot_ui], [chatbot_ui])
    example_button.click(buttons.generate_example, [chatbot_ui], [chatbot_ui])
    use_history_button.click(use_saved_history, [], [chatbot_ui])

# Lançar a interface Gradio com o cleanup após a sessão
demo.launch(share=True, auth=("joao", "ferreira"))  # Limpeza após o fechamento da interface
