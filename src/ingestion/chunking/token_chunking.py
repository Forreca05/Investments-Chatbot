from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document
from src.ingestion.chunking.chunking_base import ChunkingBase
from llama_index.core import SimpleDirectoryReader




class TokenChunking(ChunkingBase):
    def __init__(self):
        self.DEFAULT_CHUNK_SIZE=500
        self.DEFAULT_CHUNK_OVERLAP=100
    
    def _text_splitter(self):
        print("Running token chunker...")        
        splitter = TokenTextSplitter(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=self.DEFAULT_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(self.text)
        return chunks
    
    def get_chunks_lenght(self):
        """Returns the number of chunks for the document."""
        return len(self.chunks)
    
    def get_chunks_from_text(self, text:str) -> list:
        self.text = text
        self.chunks= self._text_splitter()
        return self.chunks
    
    def get_metadata(self, node):
        raise NotImplementedError
 


def text_to_chunks(text: str, max_chunk_size: int = 500):
    """Divide o texto em pedaços, agrupando parágrafos e respeitando o tamanho máximo."""
    paragraphs = text.split("\n")  # Divide por parágrafos
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for paragraph in paragraphs:
        paragraph_size = len(paragraph.split())
        if current_chunk_size + paragraph_size <= max_chunk_size:
            current_chunk.append(paragraph)
            current_chunk_size += paragraph_size
        else:
            chunks.append("\n".join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_size = paragraph_size
    
    # Adiciona o último pedaço
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks

