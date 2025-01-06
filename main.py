import chromadb
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os


#Provide Environment Variables
load_dotenv()
CHROMADB_HTTPS_ADDRESS=os.getenv('CHROMADB_HTTPS_ADDRESS')
CHROMADB_PORT=int(os.getenv('CHROMADB_PORT'))
CHROMADB_COLLECTION=os.getenv('CHROMADB_COLLECTION')
EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL')
LLM_MODEL=os.getenv('LLM_MODEL')
RETRIEVER_K_NUMBER=int(os.getenv('RETRIEVER_K_NUMBER'))
RETRIEVER_RELEVANCE_SCORE=float(os.getenv('RETRIEVER_RELEVANCE_SCORE'))
OLLAMA_HOST=os.getenv('OLLAMA_HOST')



def get_collection(client,collection_name:str) -> chromadb.Collection:
    collection = client.get_or_create_collection(name=collection_name)
    return collection



def get_metadata(collection:chromadb.Collection,metadata_field:str,**kwargs) ->list:

    if kwargs.get('filter_dict') == None:
        doc_metadata = collection.get(include=["metadatas"])["metadatas"]
    else:
        filter_dict=kwargs.get('filter_dict')
        if filter_dict == {'Topic':{'$in':[]}}:
            doc_metadata = collection.get(include=["metadatas"])["metadatas"]
        else:
            doc_metadata = collection.get(include=["metadatas"],where=filter_dict)["metadatas"]

    metadata_values = list(set(list(map(lambda x: x[metadata_field],doc_metadata))))
    return metadata_values

def retriever(vector_store:Chroma, prompt: str,k_number:int,score_threshold:float,filter_dict:dict) -> list:
    # Perform the vector search query on the Chroma vector store
    results = vector_store.similarity_search_with_relevance_scores(
        query=prompt,
        k=k_number,
        score_threshold = score_threshold,
        filter = filter_dict
   )
    return results



def main():

    llm = Ollama(model=LLM_MODEL,base_url=OLLAMA_HOST)
    #llm = ChatOpenAI(api_key=OPENAI_API_KEY,model_name=LLM_MODEL)
    embeddings = OllamaEmbeddings(
        #api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST
)   
    
    chroma_client = chromadb.HttpClient(
        host=CHROMADB_HTTPS_ADDRESS,
        port=CHROMADB_PORT
        )
    collection_name = CHROMADB_COLLECTION
    
    vector_store = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function= embeddings
)
    
    #retriever = vector_store.as_retriever()

    st.set_page_config(page_title="Research Assistant",
                       page_icon=":books:")
    st.header("Chat with your Scientific Papers :books:")

    collection = get_collection(client=chroma_client,collection_name=collection_name)

    prompt_template = ChatPromptTemplate.from_messages(
    [
       ("system", """ 
You are a scientific research assistant that is here to help university students with their homework. Your Name is Prastoo. 
Contextual data: 
{context}

Insturctions: 
- Contextual data comes in the form of a langchain document with Title and Author in the metadata. 
- In your answer stay as close as possible to the wording of the contextual data and cite it in APA 6 if possible             
- If you are unable to answer the question by the provided contextual data, reply 'I dont know - reach out to your professor for further information or check a different topic'
- Make use of Markdown to highlight parts that are important for the students
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Student Question:{input}"),  
    ]
)



    chain = prompt_template | llm | StrOutputParser()


    with st.sidebar:
        st.subheader('Topic Selection')
        selected_topic = st.multiselect(label="Select one (or multiple) topics that you would like to investigate",options=get_metadata(collection=collection,metadata_field="Topic"),default=get_metadata(collection=collection,metadata_field="Topic")[0], help="Topics that you select contain multiple documents. Hence questions that you ask will take place in the defined context", placeholder="Select a topic to chat with")
        st.subheader('Document Selection (optional)')
        selected_documents = st.multiselect(label="Select one (or multiple) topics that you would like to investigate",options=get_metadata(collection=collection,metadata_field="Title",filter_dict={'Topic':{'$in':selected_topic}}),default=get_metadata(collection=collection,metadata_field="Title",filter_dict={'Topic':{'$in':selected_topic}}), help="Topics that you select contain multiple documents. Hence questions that you ask will take place in the defined context", placeholder="Select a topic to chat with")
        if st.button('restart conversation'):
            st.session_state["chat_history"] = []

    # initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        if type(message) is HumanMessage:
            with st.chat_message("user"):
                st.markdown(message.content)
        elif type(message) is AIMessage:
            with st.chat_message("assistant"):
                st.markdown(message.content)
        else: 
            with st.chat_message("assistant"):
                st.markdown("oops")

    if prompt := st.chat_input('Ask your question about this paper'):
        with st.chat_message("user"):
            st.markdown(prompt)
        if len(selected_topic) > 0 and len(selected_documents) > 0:
            filter_dict={"$and":[{"Topic":{"$in":selected_topic}},{"Title":{"$in":selected_documents}}]}
        elif len(selected_topic) > 0 and len(selected_documents) == 0:
            filter_dict={"Topic":{"$in":selected_topic}}
        elif len(selected_topic) == 0 and len(selected_documents) > 0:
            filter_dict={"Title":{"$in":selected_documents}}

                    
        context = retriever(vector_store=vector_store,prompt=prompt,k_number=RETRIEVER_K_NUMBER,score_threshold=RETRIEVER_RELEVANCE_SCORE,filter_dict=filter_dict)
        st.session_state["chat_history"].append(HumanMessage(content=prompt))
         

        with st.chat_message("assistant"):
            ai_response = st.write_stream(chain.stream({"input": prompt,"context":context,  "chat_history": st.session_state["chat_history"]}))

        st.session_state["chat_history"].append(AIMessage(content=ai_response))



if __name__ == '__main__':
    main()
