# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import configparser
import getpass
import os
import streamlit as st

config = configparser.ConfigParser()
config.read("environ.conf")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.get("Environment Variables", "GOOGLE_APPLICATION_CREDENTIALS")
os.environ["DOC_FOLDER_PATH"] = config.get("Environment Variables", "DOC_FOLDER_PATH")
os.environ["LANGCHAIN_TRACING_V2"] = config.get("Environment Variables", "LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = config.get("Environment Variables", "LANGCHAIN_API_KEY")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_model():
    model = VertexAI(model_name="gemini-pro", temperature=0.8)
    return model


def load_docs_into_vectorstore():
    loader = DirectoryLoader(os.environ.get("DOC_FOLDER_PATH"), glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore


def set_up_prompt_template():
    template = """You are a travel agent. Answer the questions based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def get_chain():
    model = get_model()
    vectorstore = load_docs_into_vectorstore()
    retriever = vectorstore.as_retriever()
    prompt = set_up_prompt_template()
    output_parser = StrOutputParser()
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | model | output_parser
    return chain, vectorstore


def get_response_for_question(question):
    chain, vectorstore = get_chain()
    response = chain.invoke(question)
    for chunk in response:
        print(chunk, end="", flush=True)
    return response, vectorstore


def create_streamlit():
    st.title("Ask AI: Traveling in Spain")
    with st.form("travel_form"):
        question = st.text_area("Ask your question here:", "What's our travel plan for May 19th, 2024?")
        submitted = st.form_submit_button('Ask AI Travel Agent')
        if submitted:
            response, vectorstore = get_response_for_question(question)
            st.info(response)
            vectorstore.delete_collection()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #ask_question = "What's our travel plan for May 21st, 2024?"
    #get_response_for_question(ask_question)
    create_streamlit()
