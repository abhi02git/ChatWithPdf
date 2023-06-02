import os
from apikey import apikey

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain


import streamlit as st
from PyPDF2 import PdfReader


#setting up the openai api
os.environ['OPENAI_API_KEY'] = apikey

#app environment
st.title('🦜🔗Chat with PDF')


#upload pdf
pdf = st.file_uploader("Upload your pdf", type="pdf")


#extract the text
if pdf:
    pdf_reader = PdfReader(pdf)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()
        
        #split into chunks
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        
        chunks = text_splitter.split_text(text)
        
        #create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        user_question = st.text_input("Ask me questions from your pdf", key= "user_query")
        
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            # st.write(docs)
            
            llm = OpenAI(temperature = 0)
            
            chain = load_qa_chain(llm, chain_type = "map_reduce", verbose = False)
            response = chain.run(input_documents = docs, question = user_question, verbose = False)
            
            st.write(response)