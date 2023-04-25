from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma
from template import STUFF_PROMPT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from typing import List, Dict, Any
import streamlit as st
from openai.error import AuthenticationError
from yt_loader import YoutubeLoader


def ingest_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    return transcript



def text_to_docs(text: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(text)
    return docs


def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at https://platform.openai.com/account/api-keys."
        )
    else:
        # Embed the chunks
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.get("OPENAI_API_KEY"))  # type: ignore
        index = Chroma.from_documents(docs, embeddings)

        return index



def search_docs(index: VectorStore, query: str) -> List[Document]:
    docs = index.similarity_search(query, k=5)
    return docs


def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:

    chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=st.session_state.get("OPENAI_API_KEY")), chain_type="stuff", prompt=STUFF_PROMPT)  # type: ignore

    answer = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )

    return answer

