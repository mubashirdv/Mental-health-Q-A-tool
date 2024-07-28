from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from datasets import load_dataset
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.schema.output_parser import OutputParserException

load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.5)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    # Load the dataset
    dataset = load_dataset('ZahrizhalAli/mental_health_conversational_dataset')
    data = pd.DataFrame(dataset['train'])
    data = data['text'].str.split("\n<ASSISTANT>:", expand=True)
    data.columns = ['input', 'output']
    # Removing '<HUMAN> ' from the 'input' column
    data['input'] = data['input'].str.replace("<HUMAN>: ", "")
    # Check for rows with null values in 'output' and re-split them
    for index, row in data[data['output'].isnull()].iterrows():
        # Re-split the 'input' column using '/n<ASSISTANT>'
        split_row = row['input'].split("/n<ASSISTANT>")
        if len(split_row) == 2:
            data.at[index, 'input'] = split_row[0].strip()
            data.at[index, 'output'] = split_row[1].strip()
    # save the clean data
    data.to_csv('clean_data.csv')
    
    # Convert DataFrame rows to Document objects
    documents = [Document(page_content=row['input'], metadata={'output': row['output']}) for index, row in data.iterrows()]

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=documents, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """
    You are an expert assistant with extensive knowledge in mental health. Use the following context to answer the question as thoroughly and accurately as possible.

    If the context does not contain the answer, say "I don't know."

    Examples:
    Context: "Anxiety disorders are characterized by significant feelings of anxiety and fear. Anxiety is a worry about future events, and fear is a reaction to current events."
    Question: "What are anxiety disorders?"
    Answer: "Anxiety disorders are characterized by significant feelings of anxiety and fear. Anxiety is a worry about future events, and fear is a reaction to current events."

    Context: "Cognitive-behavioral therapy (CBT) is effective for treating anxiety disorders. It focuses on changing unhelpful cognitive distortions and behaviors."
    Question: "How is anxiety treated?"
    Answer: "Anxiety can be treated with cognitive-behavioral therapy (CBT), which focuses on changing unhelpful cognitive distortions and behaviors."

    Context: {context}

    Question: {question}
    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    try:
        response = chain("what are the causes of anxiety?")
        print(response)
    except OutputParserException as e:
        print(f"OutputParserException: {e}")
    except IndexError as e:
        print(f"IndexError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
