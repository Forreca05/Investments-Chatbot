from src.ingestion.loaders.loaderBase import LoaderBase
from docx import Document
from pathlib import Path

class LoaderDOCX(LoaderBase):

    def __init__(self, filepath:str):
        self.filepath=filepath

    def extract_metadata(self):
        from zipfile import ZipFile

        metadata = {}

        try:
            with ZipFile(self.filepath, 'r') as docx_zip:
                core_properties = docx_zip.read('docProps/core.xml').decode('utf-8')
                import xml.etree.ElementTree as ET
                root = ET.fromstring(core_properties)
                for elem in root.findall('.//{http://purl.org/dc/elements/1.1/}creator'):
                    metadata['author'] = elem.text
                for elem in root.findall('.//{http://purl.org/dc/elements/1.1/}title'):
                    metadata['title'] = elem.text
                for elem in root.findall('.//{http://purl.org/dc/elements/1.1/}subject'):
                    metadata['subject'] = elem.text
        except Exception as e:
            print(f"Erro ao extrair metadados: {e}")
            return False
        
        return metadata if self.all_keys_have_values(metadata) else False

    
    def extract_text(self):
        doc = Document(self.filepath)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    def extract_chunks(self):
        raise NotImplementedError 
    
    def all_keys_have_values(self, metadata, value_check=lambda x: x is not None and x != ''):
        return all(value_check(value) for value in metadata.values())