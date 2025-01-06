
**Setting up the environment Variables**
Following enviorment variables are required and to be defined in the .env file: 

- EMBEDDING_MODEL: This defines the embedding model that you want to use from Ollama. Please make sure that you have pulled the model from Ollama. You can find a list of embedding models [HERE](https://ollama.com/search?c=embedding). *Example Value ->* "mxbai-embed-large" 
- OLLAMA_HOST: This defines the host of your Ollama instance. Per default it is "127.0.0.1:11434", but after changing the OLLAMA_HOST enviornment variable of your system it should point to your localhost, otherwise the Docker Container will no be able to access it. Further information on the same can be found [HERE](https://www.restack.io/p/ollama-answer-bind-to-0-0-0-0-cat-ai). *Example Value ->* "192.168.0.27:11434" 
- CHROMADB_HTTPS_ADDRESS: This is the address of your Chroma Docker container, that is used for the ChromaHttpClient within the streamlit App. When running the container (via docker run) for example with --port 8000:8000 the database will be exposed to your localhost and port that you defined in afore mentioned argument.  *Example Value ->* "192.168.0.27"
- CHROMADB_PORT: The port of the host on which the ChromaDb is running. *Example Value ->* 8000
- CHROMADB_COLLECTION=This is the name of the collection that you want to create inside of your chroma datbase. Further information of collections can be found [HERE](https://cookbook.chromadb.dev/core/collections/). *Example Value ->* "scientific_papers"
- RETRIEVER_K_NUMBER: The maximum number of documents to retrieve. *Example Value ->*  10
- RETRIEVER_RELEVANCE_SCORE: Defines the similartiy percentage at which a document will be qualified as not relevant. The value ranges from 0 to 1 and is a decimal. *Example Value ->*  0.2


**Run App locall in Streamlit inside of venv** 

The below steps should work fine if you are using python:3.11 or higher

Required Steps:
- python -m venv .venv 
- .venv/scripts/activate
- pip install -r requirements.txt
- streamlit run main.py       


**Build docker image, e.g. after doing changes**
docker build -t thesis_rag .

**Run the docker image py passing the env variables**

docker run  --env=EMBEDDING_MODEL=mxbai-embed-large --env=OLLAMA_HOST=http://192.168.0.27:11434 --env=CHROMADB_HTTPS_ADDRESS=192.168.0.27 --env=CHROMADB_PORT=8000 --env=CHROMADB_COLLECTION=scientific_papers --env=RETRIEVER_K_NUMBER=20 --env=RETRIEVER_RELEVANCE_SCORE=0.2 --env=LLM_MODEL=llama3.2:3b --mount type=bind,src=F:\Dokumente\rag_logs,dst=/thesis_rag/logs  -p 8504:8504 -d rag_thesis:latest

Important: Be sure that you provide the correct path for a bind mount to make the conversation logs available externally of the docker container. For that provide the folder path on your host machine, to which you want the logs to be written in the src argument, and specify the folder inside of the docker container. The logs folder will be inside of the folder, which is named like the name of the image that you created in the docker build command. Hence /--YOUR DOCKER IMAGE NAME--/logs

**Prequisites**
- Make sure to set OLLAMA_HOST environment variable to 0.0.0.0, otherwise the dockercontainer will not be able to access the ollama service
- Make sure that you are running version 0.5.23 of ChromaDB. There have been compatability issues between langchain Python libraries and newever chromadb versions.. [Guide to installing Chroma as a Docker Container](https://docs.trychroma.com/production/containers/docker)


