from src.ingestion.loaders.loaderBase import LoaderBase

class LoaderCSV(LoaderBase):

    def __init__(self,filepath:str):
        self.filepath=filepath
    
    def extract_metadata(self):
        raise NotImplementedError
    
    def extract_text(self):
        raise NotImplementedError("Implement LoaderCSV.extract_text() to extract text from a CSV file.")            