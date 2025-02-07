import os
from src.services.vectorial_db.faiss_index import FAISSIndex
from src.ingestion.loaders.loader import Loader

DATA_FOLDER = 'data'

def ingest_files_data_folder(index: FAISSIndex):
    """Ingests all files in the data folder into the FAISS index."""
    
    # Obtém apenas arquivos (ignora diretórios)
    files = filter(lambda f: os.path.isfile(os.path.join(DATA_FOLDER, f)), os.listdir(DATA_FOLDER))

    for file in files:
        file_path = os.path.join(DATA_FOLDER, file)
        extension = file.split(".")[-1]
        
        loader = Loader(extension=extension, filepath=file_path)
        text = loader.extract_text()
        
        print(f"Ingesting {file}")
        index.ingest_text(text=text)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    from src.services.models.embeddings import Embeddings
    
    embeddings = Embeddings()
    index = FAISSIndex(embeddings=embeddings.get_embeddings)    
    ingest_files_data_folder(index)
    index.save_index()
