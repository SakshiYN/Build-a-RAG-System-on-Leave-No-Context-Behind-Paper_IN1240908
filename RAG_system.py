import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from IPython.display import Markdown as md

# Setup API Key
f = open('C:/Users/saksh/OneDrive/Desktop/GenAI_App/genai_app_1/keys/.gemini_api_key.txt')
GOOGLE_API_KEY = f.read()

chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-pro-latest")

st.header("🕵️🔍The RAG Q&A System: Unveiling Insights from 'Leave No Context Behind📜")

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

print(type(retriever))

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    Welcome to the RAG System for "Leave No Context Behind" Paper. 
    As a RAG AI, your task is to provide insightful answers based on the context provided."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Please provide an answer to the question below, considering the context of the "Leave No Context Behind" paper:
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

output_parser = StrOutputParser()

from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

user_question = st.text_input("Enter your question here:")
if st.button("Ask"):
    response = rag_chain.invoke(user_question)
    st.markdown(response)
