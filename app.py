### 로컬 실행시에는 주석 처리 ########################3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#########################################################

import streamlit as st
import requests
import pickle
import random
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings



load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# embedding_model = HuggingFaceEmbeddings("BAAI/bge-m3")

model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )
my_llm = ChatGroq(temperature=0, model = "llama-3.1-8b-instant")

st.set_page_config(page_title="HDX Project_001", layout="wide")
st.markdown(
            """
        <style>
            .st-emotion-cache-janbn0 {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()



db_path = "./db/chroma_jujaewon_01"
my_vectorstore = Chroma(collection_name="collection_01", persist_directory=db_path, embedding_function=embedding_model)
my_retriever = my_vectorstore.as_retriever(search_type="mmr",search_kwargs={'k': 3, 'fetch_k': 50})


def quick_rag_chat(query, retriever, model):

    # if json_style:
    #     system_prompt = ('''
    # You are an assistant for question-answering tasks. 
    # Use the following pieces of retrieved context to answer the question. 
    # If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

    # {context}
    # Please provide your answer in the following JSON format: 
    # {{
    # "answer": "Your detailed answer here",\n
    # "keywords: [list of important keywords from the context] \n
    # "sources": "Direct sentences or paragraphs from the context that support your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
    # }}
    # The JSON must be a valid json format and can be read with json.loads() in Python. Answer:
    #                     ''')
    
    # else: 
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ]
        )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": f"{query}"})
    return response["context"], response["answer"]




if "messages" not in st.session_state:   st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "time_delta" not in st.session_state:   st.session_state.time_delta = ""
if "doc_list" not in st.session_state:   st.session_state.doc_list = list()
if "check_monitoring" not in st.session_state:   st.session_state.check_monitoring = False  


if __name__ == "__main__":
    st.title("HDX Project 001")
    st.markdown("---")


    query = st.chat_input("Say something")
    with st.spinner("Processing..."):
        if query:
            start_time = datetime.now()
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.doc_list, answer = quick_rag_chat(query, my_retriever, my_llm)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            end_time = datetime.now()
            st.session_state.time_delta = calculate_time_delta(start_time, end_time)
        else: pass

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
    with st.container(border=True):
        for doc in st.session_state.doc_list:
            with st.container(border=True):
                st.markdown(doc.page_content)
                st.markdown(f":green[{doc.metadata}]")

    if st.session_state.time_delta: 
        st.warning(f"⏱️ TimeDelta(Sec) : {st.session_state.time_delta}")

