from src.ingestion.loaders.loaderBase import LoaderBase
from bs4 import BeautifulSoup
from pathlib import Path
import os

class LoaderHTML(LoaderBase):
    
    def __init__(self, filepath: str):
        self.filepath = filepath

    def extract_metadata(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
            
            metadata = {
                'title': soup.title.string if soup.title else None,
                'author': self._extract_meta_tag(soup, 'author'),
                'description': self._extract_meta_tag(soup, 'description'),
                'keywords': self._extract_meta_tag(soup, 'keywords'),
                'generator': self._extract_meta_tag(soup, 'generator'),
            }

            self.metadata = metadata

            return self.metadata if self.all_keys_have_values(metadata=self.metadata) else False
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return False

    def extract_text(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')

            # Extract the main text content, excluding script and style tags
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def extract_chunks(self):
        raise NotImplementedError

    def all_keys_have_values(self, metadata, value_check=lambda x: x is not None and x != ''):
        return all(value_check(value) for value in metadata.values())

    @staticmethod
    def _extract_meta_tag(soup, name):
        """Helper method to extract content from a meta tag."""
        tag = soup.find('meta', attrs={'name': name})
        return tag['content'] if tag and 'content' in tag.attrs else None
