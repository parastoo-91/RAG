FROM python:3.11

WORKDIR /thesis_rag

ENV LLM_MODEL=-
ENV EMBEDDING_MODEL=-
ENV CHROMADB_HTTPS_ADDRESS=-
ENV CHROMADB_PORT=0
ENV CHROMADB_COLLECTION=-
ENV RETRIEVER_K_NUMBER=0
ENV RETRIEVER_RELEVANCE_SCORE=0
ENV RERANKER_MODEL=-

ADD main.py .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN  pip install -r requirements.txt 

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit","run", "main.py","--server.port=8504","--server.address=0.0.0.0"]
