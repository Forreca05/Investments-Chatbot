from src.ingestion.loaders.loaderBase import LoaderBase
import html2text

class LoaderHTML(LoaderBase):

    def __init__(self,filepath:str):
        self.filepath=filepath
    

    def extract_metadata(self):
        raise NotImplementedError
    
    def extract_text(self):
        raise NotImplementedError("Implement LoaderHTML.extract_text() to extract text from a HTML file.")
            