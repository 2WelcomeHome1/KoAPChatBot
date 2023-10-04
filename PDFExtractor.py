import pdfplumber
import requests
from pathlib import Path

def download_pdf(url: str, file_name:str):
    with open(f'{file_name}.pdf', 'wb') as f:
        f.write(requests.get(url).content)

def pdf_to_text (file_path: str):
    if Path(file_path).is_file() and Path(file_path).suffix == '.pdf':
    
        with pdfplumber.PDF(open(file=file_path, mode= 'rb')) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        text = ''.join (pages)
        
        with open('text1.txt', 'w') as file: ## С сохранением разметки 
            file.write(text)
            
        text = text. replace("\n",'') 
        
        with open('text2.txt', 'w') as file: ## В одну строку
            file.write(text)
    else:
        raise 'File not exists, check the file path!'

def run(url: str, file_name: str):
    download_pdf(url, file_name)
    pdf_to_text('{}.pdf'.format(file_name))



run('https://www.mos.ru/upload/documents/files/8414/ZakongMoskviot21112007N45(redot22052019)KoAPgMoskvi.pdf', 'KoAP')
