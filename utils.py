from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    
)
from langchain.vectorstores import Chroma
from template import STUFF_PROMPT
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain import OpenAI, Cohere
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS, VectorStore
import docx2txt
from typing import List, Dict, Any
import re
import numpy as np
from io import StringIO
from io import BytesIO
import streamlit as st
from pypdf import PdfReader
from openai.error import AuthenticationError
from yt_loader import YoutubeLoader
import textwrap

from youtube_transcript_api import YouTubeTranscriptApi

def ingest_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    #loader = YoutubeLoader(video_url, True)
    #docs = loader.load()
    #transcript = YouTubeTranscriptApi.get_transcript(video_id=video_url)    

    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #docs = text_splitter.split_documents(transcript)

    #db = Chroma.from_documents(docs, embeddings)
    return transcript



# Example usage:



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
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""
    # Search for similar chunks
    docs = index.similarity_search(query, k=5)
    return docs


def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=st.session_state.get("OPENAI_API_KEY")), chain_type="stuff", prompt=STUFF_PROMPT)  # type: ignore

    # Cohere doesn't work very well as of now.
    # chain = load_qa_with_sources_chain(Cohere(temperature=0), chain_type="stuff", prompt=STUFF_PROMPT)  # type: ignore
    answer = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )


    


    return answer



def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])