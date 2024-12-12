import csv
from src.ingestion.loaders.loaderBase import LoaderBase

class LoaderCSV(LoaderBase):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def extract_metadata(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                headers = next(reader, None)  # Get the header row (if any)
                row_count = sum(1 for _ in reader)  # Count the rows
                
            metadata = {
                "columns": headers if headers else [],
                "num_columns": len(headers) if headers else 0,
                "num_rows": row_count,
            }
            
            self.metadata=metadata
            return self.metadata if self.all_keys_have_values(metadata=self.metadata) else False
    
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return False

    def extract_text(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                text_data = "\n".join([", ".join(row) for row in reader])
            return text_data
        
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def extract_chunks(self):    
        try:
            chunks = []
            with open(self.filepath, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                chunk_size = 100  # Define chunk size (e.g., 100 rows per chunk)
                chunk = []
                for i, row in enumerate(reader, start=1):
                    chunk.append(row)
                    if i % chunk_size == 0:
                        chunks.append(chunk)
                        chunk = []
                if chunk:  # Add any remaining rows
                    chunks.append(chunk)
            return chunks
        except Exception as e:
            print(f"Error extracting chunks: {e}")
            return []
