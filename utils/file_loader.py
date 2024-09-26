from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain import text_splitter
from langchain_core.documents import Document
from PyPDF3 import PdfFileReader
import logging
from uuid import uuid4
import ollama
import chromadb
from dotenv import load_dotenv
import os

class chunker:
    def __init__(self, ChunkSize, ChunkOverlap):
       self.ChunkSize = ChunkSize
       self.ChunkOverlap = ChunkOverlap
       self.ch_logger = logging.getLogger("__name__")
       self.ch_logger.setLevel(level=logging.INFO)
       


    def __merge_dicts(self, dict1,*args):
        """
        can be used to merge multiple dicts
        """
        fin_dict = dict1
        for i in args:
           fin_dict.update(i)
        return fin_dict
     
    def __extract_pdf_metadata(self,file_path: str):
       """
       Extracts metadata from the PDF file.
       """
       with open(file_path, 'rb') as pdf_file:
         pdf_reader = PdfFileReader(pdf_file)
         metadata = pdf_reader.getDocumentInfo()
         return {key[1:]: value for key, value in metadata.items()}  # Remove leading "/" from keys
       
    def pdf_load(self,FilePath:str,**kwargs):
       """
       FilePath = Path to the PDF File
       ChunkSize = Size of the Chunks
       ChunkOverlap = Overlap of the Chunks
       *kwargs = further metadata to be added
       """
       pdf =  PyPDFLoader(file_path=FilePath)
       pdf_content = pdf.load_and_split(text_splitter=text_splitter.RecursiveCharacterTextSplitter(
       chunk_size=self.ChunkSize, chunk_overlap=self.ChunkOverlap, keep_separator=True))
       pdf_metadata = self.__extract_pdf_metadata(FilePath)
       update_dict = {}
       self.ch_logger.info("Working on file")
       for key,value in kwargs.items():
          update_dict[key] = value
       doc_list =  list(map(lambda doc: Document(page_content= doc.page_content , metadata= self.__merge_dicts(doc.metadata,pdf_metadata, update_dict))  ,pdf_content))
       
       return doc_list

#class vectorizer:
#    
#    def __init__():
#        load_dotenv()
#
#    @staticmethod
#    def add_documents(doc_list:list,model:str, collection:chromadb.Collection):
#        #logger.info('Adding document list to the collection')
#        print('adding documents')
#        for d in doc_list:
#            response = ollama.embeddings(model=os.getenv("EMBEDDING_MODEL"), prompt=d.page_content)
#            embedding = response["embedding"]
#            collection.add(
#                ids=[str(uuid4())],
#                embeddings=[embedding],
#                documents=[d.page_content],
#                metadatas=d.metadata
#            )
  