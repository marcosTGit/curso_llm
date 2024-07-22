# pip install streamlit langchain langchain-community freeze-requirements faiss-cpu langchain-openai pypdf colorama
from dotenv import load_dotenv
load_dotenv()
import os
# from colorama import init, Fore, Style
# init(autoreset=True)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI

# from openai import OpenAI
import streamlit as st

# 00000000000000000000000000000000000000000000000000000000000000000000000000
# 00000000000000000000000000000000000000000000000000000000000000000000000000
# 00000000000000000000000000000000000000000000000000000000000000000000000000
# 00000000000000000000000000000000000000000000000000000000000000000000000000
# procesamos el documento 

loader = PyPDFLoader("./carta.pdf")
documento = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)

document_chunks = text_splitter.split_documents(documento)


# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # Cargamos los embeddings en una base de datos vectorial
#  # !pip install faiss-cpu -Uq

# openai_api_key = os.getenv("OPENAI_API_KEY")
# embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
# stored_embeddings = FAISS.from_documents(document_chunks, embeddings_model)

# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # Creamos una Cadena de Preguntas y Respuetas con Recuperación

# llm = OpenAI(api_key=openai_api_key)

# QA_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=stored_embeddings.as_retriever()
# )

# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000
# # # 00000000000000000000000000000000000000000000000000000000000000000000000000




# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c
# 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c



with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Pizzeria la Delici<a")
st.caption("🚀 chatbot - OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hola enque puedo ayudarte"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("ingresa tu OpenAI API key para continuar.")
        st.stop()

    # openai_api_key = os.getenv("OPENAI_API_KEY")
    try:
        print("000000000000000000000000000000000000000000000000000000")
        print("000000000000000000000000000000000000000000000000000000")
        print("000000000000000000000000000000000000000000000000000000")
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
        stored_embeddings = FAISS.from_documents(document_chunks, embeddings_model)

        llm = OpenAI(api_key=openai_api_key)

        QA_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=stored_embeddings.as_retriever()
        )

        # client = OpenAI(api_key=openai_api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        # msg = response.choices[0].message.content
        # # 00000000000000000000000000000000
        # # 00000000000000000000000000000000
        resp=QA_chain.invoke(prompt)
        msg=str(resp["result"])
        # # 00000000000000000000000000000000
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    except:
        st.info("Al parecer tu OpenAI API key no es correcta o valida, verifica los datos ingresados.")
        st.stop()
        