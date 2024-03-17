from pathlib import Path
import os
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

root_dir = Path(Path.home(), "Desktop\Misc\Tech Hackathon 2024")
base_dir = Path(root_dir, 'sustainability-benchmarking')

dataset_dir = Path(base_dir, 'dataset')

chroma_persist_directory = Path(root_dir, 'output', 'chroma_db')

# List of companies
list_of_companies = ['ESCO', 'Regal Rexnord', 'Watts', 'AMETEK', 'Crane Co', 'Enpro', 'Rockwell Automation', 'Chart', 'Columbus Mckinnon', 'Graco', 'IDEX', 'Franklin Electric', 'Nordson', 'SPX']

## List of dataset
# companies_pdf = {}
# for org, filename in zip(list_of_companies, os.listdir(dataset_dir)):
#     companies_pdf[org] = filename
# print(companies_pdf)
companies_pdf = {'ESCO': '2022_ESCO_ESG_Report.pdf',
                 'Regal Rexnord':'2022-Regal-Rexnord-Sustainability-Report.pdf',
                 'Watts':'630570.pdf',
                 'AMETEK': 'AMETEK_SustainabilityReport_2021.pdf',
                 'Crane Co': 'Crane-Co_PSE_2022_2.27.23_FINAL.pdf',
                 'Enpro': 'enpro_sustainability23_final.pdf',
                 'Rockwell Automation ': 'esap-br032_-en-p.pdf',
                 'Chart': 'ESG2022.pdf',
                 'Columbus Mckinnon':'fy23-columbus-mckinnon-csr-report.pdf',
                 'Graco': 'Graco-ESG-Report.pdf',
                 'IDEX': 'IDEX_SustainabilityReport2022_accessible_version.pdf',
                 'Franklin Electric': 'M1952_Sustainability_Report_2023.pdf',
                 'Nordson': 'NordsonESGReport_updated110222.pdf',
                 'SPX': 'SPX.pdf'
                 }

# create the open-source embedding function
embeddings_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")