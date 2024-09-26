import chromadb
import os
import ollama
from dotenv import load_dotenv
from utils.file_loader import chunker
from utils.vectorizer import add_documents
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder




def get_collection(client:chromadb.PersistentClient,collection_name:str) -> chromadb.Collection:
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def get_pdf_titles(collection:chromadb.Collection) ->list:
    doc_metadata = collection.get(include=["metadatas"])["metadatas"]
    doc_titles = list(set(list(map(lambda x: x['Title'],doc_metadata))))
    return doc_titles

def process_files(pdf_file_path:str, collection:chromadb.Collection) -> None:
    to_process_list = os.listdir(pdf_file_path)
    for d in to_process_list: 
        chunk = chunker(ChunkSize=75000,ChunkOverlap=600)
        doc_list = chunk.pdf_load(FilePath=  pdf_file_path  + "/" + d )
        add_documents(doc_list=doc_list,model=os.getenv("EMBEDDING_MODEL"),collection=collection)

def retriever(prompt:str,doc_title:str,collection:chromadb.Collection, n_results:int)->list:
    response = ollama.embeddings(
    prompt=prompt,
    model=os.getenv("EMBEDDING_MODEL")
)
    results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=n_results,
    where={"Title":doc_title}
)
    docs = results['documents'][0]
    return docs

def main():
    llm = Ollama(model="llama3")


    load_dotenv()
    pdf_file_path = "data/pdf/00_to_process/"
    chroma_client = chromadb.PersistentClient()
    collection_name = "pdf_collection"

    st.set_page_config(page_title="Chat with your scientific paper",
                       page_icon=":books:")
    st.header("Chat with your scientific paper :books:")
    collection = get_collection(client=chroma_client,collection_name=collection_name)

    prompt_template = ChatPromptTemplate.from_messages(
    [

       # SystemMessage(content= """
       #               You are a scientific research assistant that is here to help university students with their homework. Your Name is Parastoo. 
#
       #               Contextual data: 
       #               {context}
#
       #               In your answer stay as close as possible to the wording of the contextual data, if possible quote it. If you are unable to answer the question by the provided contextual data, reply 'I dont know'. Return the contextual data before your response. 
#
       #               """),
       ("system", """ 
You are a scientific research assistant that is here to help university students with their homework. Your Name is Parastoo. 
Contextual data: 
{context}
In your answer stay as close as possible to the wording of the contextual data, if possible quote it. If you are unable to answer the question by the provided contextual data, reply 'I dont know'. Return the contextual data before your response. The contextual data stems from a scientific research paper called {doc_title}.
 """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Student Question:{input}"),
       # ("Context", "Context provided from external documents:{context}")
        
    ]
)


    chain = prompt_template | llm 


    with st.sidebar:
        st.subheader('Document Selection')
        document_title = st.selectbox("Select the document you would like to chat with",get_pdf_titles(collection=collection),index=None, placeholder="Select a document to chat with")
        if st.button('Process newly added files'):
            with st.spinner("Processing"):
                process_files(pdf_file_path=pdf_file_path,collection = collection)
        if st.button('restart conversation'):
            st.session_state["chat_history"] = []



    #initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


    if prompt := st.chat_input('Ask your question about this paper'):
        context = retriever(prompt=prompt,doc_title=document_title,collection=collection,n_results=3)
        # add latest message to history in format {role, content}
        response = chain.invoke({"input": prompt,"context":context, "doc_title":document_title, "chat_history": st.session_state["chat_history"]})
        print(context)
        st.session_state["chat_history"].append(HumanMessage(content=prompt))
        st.session_state["chat_history"].append(AIMessage(content=response))

        for message in st.session_state["chat_history"]:
            if type(message) is HumanMessage:
                with st.chat_message("user"):
                    st.markdown(message.content)
            else: 
                with st.chat_message("assistant"):
                    st.markdown(message.content)
        


if __name__ == '__main__':
    main()