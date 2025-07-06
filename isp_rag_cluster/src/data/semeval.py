import os
import requests
import zipfile
import pandas as pd

class SemEvalAffectDataset:
    def __init__(self, zip_url, zip_path, extracted_file_path, column_mapping=None):
        """
        Initialize the dataset handler.
        :param zip_url: URL to download the ZIP file containing the dataset.
        :param zip_path: Local path to save the downloaded ZIP file.
        :param extracted_file_path: Path to the TSV file after extraction.
        :param column_mapping: Dictionary mapping original column names to desired names.
        """
        self.zip_url = zip_url
        self.zip_path = zip_path
        self.extracted_file_path = extracted_file_path
        # Default mapping; adjust these keys based on the actual dataset columns.
        self.column_mapping = column_mapping or {
            'tweet': 'text',
            'emotion': 'class',
            'arousal_score': 'arousal',
            'valence_score': 'valence'
        }
        self.data = None

    def download_zip(self):
        """Download the dataset ZIP file if not already present."""
        if not os.path.exists(self.zip_path):
            print("Downloading dataset ZIP from:", self.zip_url)
            response = requests.get(self.zip_url)
            if response.status_code == 200:
                with open(self.zip_path, 'wb') as f:
                    f.write(response.content)
                print("Download completed and saved to:", self.zip_path)
            else:
                raise Exception(f"Failed to download dataset, status code: {response.status_code}")
        else:
            print("ZIP file already exists at:", self.zip_path)

    def extract_file(self):
        """Extract the TSV file from the ZIP archive."""
        if not os.path.exists(self.extracted_file_path):
            print("Extracting file from ZIP archive...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Debug: list all files in the ZIP
                files_in_zip = zip_ref.namelist()
                print("Files in ZIP:", files_in_zip)
                # Find TSV files in the archive
                tsv_files = [f for f in files_in_zip if f.endswith('.tsv')]
                if not tsv_files:
                    raise Exception("No TSV file found in the ZIP archive.")
                # Extract the first TSV file found
                zip_ref.extract(tsv_files[0], os.path.dirname(self.extracted_file_path))
                extracted_path = os.path.join(os.path.dirname(self.extracted_file_path), tsv_files[0])
                # Rename/move extracted file to the desired path if necessary
                os.rename(extracted_path, self.extracted_file_path)
                print("Extraction completed. File saved to:", self.extracted_file_path)
        else:
            print("Extracted TSV file already exists at:", self.extracted_file_path)

    def load_data(self):
        """Load the TSV file into a pandas DataFrame."""
        if not os.path.exists(self.extracted_file_path):
            raise Exception("TSV file not found. Please ensure extraction is complete.")
        self.data = pd.read_csv(self.extracted_file_path, sep="\t")
        print("Dataset loaded with shape:", self.data.shape)

    def map_columns(self):
        """Rename DataFrame columns based on the provided mapping."""
        if self.data is None:
            raise Exception("Data not loaded. Please run load_data() first.")
        self.data.rename(columns=self.column_mapping, inplace=True)
        print("Columns after mapping:", self.data.columns.tolist())

    def get_data(self):
        """Return the processed DataFrame."""
        if self.data is None:
            raise Exception("Data not loaded. Please prepare the dataset first.")
        return self.data

    def prepare_dataset(self):
        """Full pipeline: download, extract, load, and map columns."""
        self.download_zip()
        self.extract_file()
        self.load_data()
        self.map_columns()
        return self.get_data()

# Example usage:
if __name__ == '__main__':
    # Updated Zenodo URL (as of current check)
    zip_url = "https://zenodo.org/record/1183710/files/semeval2018_task1_data.zip"
    zip_path = "semeval2018_task1_data.zip"
    extracted_file_path = "semeval2018_task1_data.tsv"
    
    # Adjust column mapping based on actual dataset structure if needed.
    column_mapping = {
        'tweet': 'text',
        'emotion': 'class',
        'arousal_score': 'arousal',
        'valence_score': 'valence'
    }
    
    dataset_handler = SemEvalAffectDataset(zip_url, zip_path, extracted_file_path, column_mapping)
    try:
        df = dataset_handler.prepare_dataset()
        print(df.head())
    except Exception as e:
        print("An error occurred:", e)