import os
import streamlit as st
import pickle
import pandas as pd
from io import StringIO
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

import openai


def generate_response(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user","content":prompt}]
    )
    message = response.choices[0].message.content
    return message

st.title("AI Assistant")

prompt = st.text_input("Enter your message:", key='prompt')

if st.button("Submit", key='submit'):
  response = generate_response(prompt)
  st.success(response)

st.title("Ask question from uploaded url")
st.sidebar.title("upload URL")

urls = []
for i in range(1):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500,model='gpt-3.5-turbo-instruct')

if process_url_clicked:

    loader = RecursiveUrlLoader(urls[0])

    main_placeholder.text("Data Loading...Started...✅✅✅")

    data = loader.load()
    #st.write(data)
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)