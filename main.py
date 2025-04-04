from typing import Iterable, List
import chromadb
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings,ChatOllama,OllamaLLM
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableGenerator
import os
import uuid
import json
from retriever import  rerank_retriever,chroma_retriever
import re

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
RERANKER_MODEL=os.getenv('RERANKER_MODEL')


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


def docs_to_json(documents:list) -> list:
    if len(documents)==0: 
        return []
    else:
        pass
    docs_json=[]
    for e in documents:
        dict_doc = dict(e)
        adj_dict =  {"Author":dict_doc["metadata"]["Author"], "Title":dict_doc["metadata"]["Title"],"Document Text":dict_doc["page_content"]}
        docs_json.append(adj_dict)
    return json.dumps(docs_json)


def write_logs(log_location:str, conversation_id:str,chat_history:list)->None:
    #Check if logs folder exists
    if os.path.exists(log_location)==False:
        os.mkdir(log_location)
    else:
        pass

    file_path = fr"{log_location}/{conversation_id}.txt"
    #create file if it doesn't exist and open it in (over)write mode [it overwrites the file if it already exists]
    log_file = open(file_path,'w+')
    log_file.write(str(chat_history))
    log_file.close()

    return None

#match left and right single quotes
single_quote_expr = re.compile(r'[\u2018\u2019]', re.U)
#match all non-basic latin unicode
unicode_chars_expr = re.compile(r'[\u0080-\uffff]', re.U)
def cleanse_unicode(s):
    if not s:
        return ""
    
    temp = single_quote_expr.sub("'", s, re.U)
    temp = unicode_chars_expr.sub("", temp, re.U)
    return temp

def docs_to_json(documents:list) -> list:
    if len(documents)==0: 
        return []
    else:
        pass
    docs_json=[]
    for e in documents:
        dict_doc = dict(e)
        adj_dict =  {"Author":dict_doc["metadata"]["Author"], "Title":dict_doc["metadata"]["Title"],"Document Text":cleanse_unicode(dict_doc["page_content"])}
        docs_json.append(adj_dict)
    return docs_json

def streaming_parse(chunks: str) -> Iterable[str]:
    # for deepseek to filter out the <think> abc </think> sections of the stream of chunks
    fin_answer = r''
    start_wr=False
    for chunk in chunks:
        eof_think=r'</think>'
        #print(chunk)
        if start_wr==True:
            return_val=chunk
        elif chunk==eof_think:
            start_wr=True
            return_val=''
        else:
            continue
        yield return_val

streaming_parse_runnable = RunnableGenerator(streaming_parse)

def main():



    llm = OllamaLLM(model=LLM_MODEL,base_url=OLLAMA_HOST,temperature=0.8)
    llm_rerank = OllamaLLM(model=RERANKER_MODEL,base_url=OLLAMA_HOST)
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
    #Check if a conversation ID exists, else create new one
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = uuid.uuid4()


    st.set_page_config(page_title="ScholarChat",
                       page_icon=":woman-surfing:")
    st.header(":woman-surfing: ScholarSurf: Your smart Course Assistant")

    collection = get_collection(client=chroma_client,collection_name=collection_name)

    prompt_template = ChatPromptTemplate.from_messages(
    [
       ("system","""
<role>
[Your primary role is to assist students by providing accurate, concise, and well-structured answers to their questions based on the provided documents. *Only consider provided documents as your knowledge*.]
[As you are an academic study assistent the user needs to unterstand where the information came from, so always cite your provided output. Do so in an academic style. If the year is not available write n.d. as the year]
<role\>

<documents>
{context}
<documents\>
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content= """
                    Student Question: {input}. Cite all your answers! If possible present the view points of different authors. 
                    
                     <instructions>
                    [You are an AI-powered academic study assistant designed to help university students with their academic studies. You have access to a curated set of documents that are in the JSON format containing the Document Text, Author and Title. Use this information to cite your answer!]
                    [Only base your reasoning on documents that you were provided. Do NOT answer the question if no documents were provided to you. Carefully consider the <role> and <documents> that were provided to you.]
                    [Use the "Author" and "Title" of the provided documents to do citations! Year will not be provided, in such cases simply write n.d.]
                    [If documents provided are not sufficient >>> Do not answer and say "I dont't know reach out to your professor or refine your question"]
                    [You can use markdown to highlight important points and to better structure your answer.]
                    <instructions\>
                     """),  
    ]
)
        
# <objective>
# [Analyze documents to clarify concepts, answer questions, and provide structured, academic responses. Be sure to always provide citations to the documents.]
# [Cite sources inline  and always include a citation section at the end. Take the source of the citation from the metadata section of the langchain document, not from within the text! These include Author and Title. Be sure to always include at least these two. ]
# [The text required for reasoning can be found within the page_content section of the documents. ]
# <objective\>
        
# <answer format>
# [Answer to the user question]
# [Citations on which the answer was based >>> Author and Title] 
# <answer format\>
                     

    chain = prompt_template | llm | streaming_parse_runnable
    #StrOutputParser()
    #streaming_parse_runnable
    


    with st.sidebar:
        st.subheader('Topic Selection')
        selected_topic = st.multiselect(label="Select one (or multiple) topics that you would like to investigate",options=get_metadata(collection=collection,metadata_field="Topic"),default=get_metadata(collection=collection,metadata_field="Topic")[0], help="Topics that you select contain multiple documents. Hence questions that you ask will take place in the defined context", placeholder="Select a topic to chat with")
        st.subheader('Document Selection (optional)')
        selected_documents = st.multiselect(label="Select one (or multiple) documents that you want to include in your conversation context",options=get_metadata(collection=collection,metadata_field="Title",filter_dict={'Topic':{'$in':selected_topic}}),default=get_metadata(collection=collection,metadata_field="Title",filter_dict={'Topic':{'$in':selected_topic}}), help="Documents that you have selected here will be taken into consideration by the LLM when you are asking a question.", placeholder="Select a topic to chat with")
        if st.button('restart conversation'):
            #Set new conversation id and empty existing conversation history
            st.session_state.conversation_id = uuid.uuid4()
            st.session_state["chat_history"] = []
        st.write(f"Chat is run by {LLM_MODEL}")


    # initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_config={"name":"user","avatar":"👩🏽‍💻"}
    assistant_config={"name":"assistant","avatar":"🐦"}

    for message in st.session_state["chat_history"]:
        if type(message) is HumanMessage:
            with st.chat_message(**user_config):
                st.markdown(message.content)
        elif type(message) is AIMessage:
            with st.chat_message(**assistant_config):
                st.markdown(message.content)
        else: 
            with st.chat_message(assistant_config):
                st.markdown("oops")

    if prompt := st.chat_input('Ask your question about this paper'):
        with st.chat_message(**user_config):
            st.markdown(prompt)
        if len(selected_topic) > 0 and len(selected_documents) > 0:
            filter_dict={"$and":[{"Topic":{"$in":selected_topic}},{"Title":{"$in":selected_documents}}]}
        elif len(selected_topic) > 0 and len(selected_documents) == 0:
            filter_dict={"Topic":{"$in":selected_topic}}
        elif len(selected_topic) == 0 and len(selected_documents) > 0:
            filter_dict={"Title":{"$in":selected_documents}}

        
       
        #context = retriever(vector_store=vector_store,prompt=prompt,k_number=RETRIEVER_K_NUMBER,score_threshold=RETRIEVER_RELEVANCE_SCORE,filter_dict=filter_dict)
        
        st.session_state["chat_history"].append(HumanMessage(content=prompt))
        
        with st.status(":wind_blowing_face: Fetching relevant documents") as retriever_status:
            context = rerank_retriever(vector_store=vector_store,llm_rerank=llm_rerank,prompt=prompt,k_number=RETRIEVER_K_NUMBER,score_threshold=RETRIEVER_RELEVANCE_SCORE,filter_dict=filter_dict)
            #context = chroma_retriever(chroma_store=vector_store,llm_rerank=llm_rerank,k_number=RETRIEVER_K_NUMBER,score_threshold=RETRIEVER_RELEVANCE_SCORE,filter_dict=filter_dict).invoke(prompt)
            context = docs_to_json(documents=context)
            if len(context) > 0:
                retriever_status.update(label=f"{len(context)} Documents were retrieved!", state="complete", expanded=False)
                #st.write(context)
                for doc in context:
                        st.write(f"""Author: 
                                 {doc["Author"]}""")
                        st.write(f"""Document Title:
                                 {doc["Title"]}""")
                        st.write(f""" Document Text:
                            {doc["Document Text"]}""")
                        st.divider()                    
            else:
                retriever_status.update(label=f"0 Documents were retrieved!", state="error", expanded=False)
                st.write(context)
            


        #with st.spinner(':wind_blowing_face: Collecting relevant documents'):
          

            print(context)
        with st.spinner(':milky_way: Thinking :pinata:'):
            with st.chat_message(name="assistant",avatar="🐦"):   
                ai_response = st.write_stream(chain.stream({"input": prompt,"context":context,  "chat_history": st.session_state["chat_history"]}))

        st.session_state["chat_history"].append(AIMessage(content=ai_response))
        #Create log version of the chat that also includes the selected documents and the selected topics of the users, to better contextualise the logs. 
        log_chat_history= {"selected_topics":selected_topic,"selected_documents":selected_documents,"chat":st.session_state["chat_history"]}
        write_logs(log_location='./logs',conversation_id=st.session_state['conversation_id'],chat_history=log_chat_history)



if __name__ == '__main__':
    main()
