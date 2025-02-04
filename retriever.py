from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import List, Dict
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank,LLMChainFilter





class chroma_retriever(BaseRetriever):
    chroma_store: Chroma
    k_number: int
    score_threshold: float
    filter_dict: Dict
    #No __init__ neededed as per following stackoverflow -> https://stackoverflow.com/questions/77914377/issue-in-setup-of-custom-retriever-in-langchain-using-baseretriever-object-has-n
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return documents based on the similarity search with parameters"""
        results = self.chroma_store.similarity_search_with_relevance_scores(
            query=query,
            k=self.k_number,
            score_threshold=self.score_threshold,
            filter=self.filter_dict
        )
        # Process results to extract documents
        docs = [doc for doc, _ in results]
        return docs



def rerank_retriever(vector_store:Chroma, llm_rerank,prompt: str,k_number:int,score_threshold:float,filter_dict:dict) -> list:
    compressor = LLMChainFilter.from_llm(llm=llm_rerank)
    #compressor = FlashrankRerank(top_n=15)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=chroma_retriever(chroma_store=vector_store,k_number=k_number,score_threshold=score_threshold,filter_dict=filter_dict))
    results = compression_retriever.invoke(prompt)
    return results

    


