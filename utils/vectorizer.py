import ollama
import chromadb
from uuid import uuid4
import logging

# Create logger
#logger = logging.getLogger('vectorizer')
#logger.setLevel(level=logging.INFO)


def add_documents(doc_list:list,model:str, collection:chromadb.Collection):
    #logger.info('Adding document list to the collection')
    print('adding documents')
    for d in doc_list:
        response = ollama.embeddings(model=model, prompt=d.page_content)
        embedding = response["embedding"]
        collection.add(
            ids=[str(uuid4())],
            embeddings=[embedding],
            documents=[d.page_content],
            metadatas=d.metadata
        )
    

