import chattemplate
import gradio as gr

from pathlib import Path
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
import CustomRecursiveCharacterTextSplitter

class KoAPBot:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.model ="model/model-q4_K.gguf"
        self.system_template = chattemplate.system_template
        self.sentence_transformer = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    def loaddatafrom_txt(self, file_path):
        if Path(file_path).is_file() and Path(file_path).suffix == '.txt': return TextLoader(file_path).load()
        else: raise 'File not exists, check the file path!'
            
    def loaddatafrom_pdf(self, file_path):
        if Path(file_path).is_file() and Path(file_path).suffix == '.pdf': return PyPDFLoader(file_path).load()
        else: raise 'File not exists, check the file path!'

    def split_text_to_chuncks(self, documents):
        return CustomRecursiveCharacterTextSplitter.split_documents(documents)
    
    def create_vectorstore(self, documents, device: str='cuda'):
        embeddings=HuggingFaceEmbeddings(model_name=self.sentence_transformer, model_kwargs={'device':device})
        return Chroma.from_documents(documents, embeddings)
    
    def ChatChain(self):
        print('Запускаем чат бота. \nОбрабатываем полученный файл')
        # documents = self.loaddatafrom_pdf(self.file_path)
        documents = self.loaddatafrom_txt(self.file_path)
        print('Разбиваем файл на сплиты')
        splitted_document = self.split_text_to_chuncks(documents)
        print('Подготавливаем эмбеддинги для модели')
        vector_store = self.create_vectorstore(splitted_document)
        
        print('Загружаем модель')
        memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        qa_prompt=PromptTemplate(template=self.system_template, input_variables=['context', 'question'])
        llm = LlamaCpp(model_path=self.model, temperature=0.1, n_ctx= 2500, top_k=30, top_p=0.9, 
                       repeat_penalty=1.1,n_gpu_layers=40, n_batch=512, n_threads=10, n_parts=1) 
        
        return RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', memory=memory,
                                           retriever=vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 10, 'lambda_mult': 0.3}),
                                           chain_type_kwargs=dict(prompt=qa_prompt)) 

class BotInterface(KoAPBot):
    def __init__(self, file_path) -> None:
        super().__init__(file_path)
        self.conversation_chain = self.ChatChain()
        print('Запускаем GUI')
        self.interface()

    def _response(self, message, history):
        print('Получил вопрос: {}'.format(message))
        return self.conversation_chain (message)['result']

    def interface(self,):
        gr.ChatInterface(self._response).launch()


BotInterface("text2.txt")