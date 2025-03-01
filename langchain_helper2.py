import os
import getpass
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import OutputParserException
from langchain.docstore.document import Document

# ✅ Load environment variables
load_dotenv()

# Get API Key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLaMA 3 model from Groq
llm = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=GROQ_API_KEY)

# Initialize instructor embeddings
model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
instructor_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb_file_path = "faiss_index"

def create_vector_db():
    dataset = load_dataset('ZahrizhalAli/mental_health_conversational_dataset')
    data = pd.DataFrame(dataset['train'])
    data = data['text'].str.split("\n<ASSISTANT>:", expand=True)
    data.columns = ['input', 'output']
    
    # Remove '<HUMAN>: ' from 'input'
    data['input'] = data['input'].str.replace("<HUMAN>: ", "", regex=True)

    # Fix null values in 'output'
    for index, row in data[data['output'].isnull()].iterrows():
        split_row = row['input'].split("\n<ASSISTANT>:")
        if len(split_row) == 2:
            data.at[index, 'input'] = split_row[0].strip()
            data.at[index, 'output'] = split_row[1].strip()

    data.to_csv('clean_data.csv')

    # Convert DataFrame rows to Document objects
    documents = [Document(page_content=row['input'], metadata={'output': row['output']}) for _, row in data.iterrows()]

    # Use FAISS vector store to store embeddings
    vectordb = FAISS.from_documents(documents=documents, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """
    You are an expert assistant with extensive knowledge in mental health. Use the following context to answer the question as accurately as possible.

    If the context does not contain the answer,only say "I don't know." nothing more than that.

    Context: {context}
    Question: {question}
    Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# if __name__ == "__main__":
#     # create_vector_db()  # Uncomment if running for the first time
#     chain = get_qa_chain()
#     try:
#         query = "What are the common causes of anxiety?"
#         response = chain.invoke({"query": query}) 
#         print(response)
#     except OutputParserException as e:
#         print(f"OutputParserException: {e}")
#     except IndexError as e:
#         print(f"IndexError: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
