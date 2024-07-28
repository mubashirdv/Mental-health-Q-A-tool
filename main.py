import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from langchain.schema.output_parser import OutputParserException

st.title("Mental Health Q&A 🌱")

btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    try:
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])
    except OutputParserException:
        st.header("Answer")
        st.write("Sorry, I can't answer this question.")
    except IndexError:
        st.header("Answer")
        st.write("Sorry, I can't answer this question.")
    except Exception as e:
        st.header("Answer")
        st.write(f"Unexpected error: {e}")
