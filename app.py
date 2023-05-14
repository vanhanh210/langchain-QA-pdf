import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    # Set up Streamlit page
    st.set_page_config(page_title="Ask your PDF")
    st.header("Questions to ask about your book💬")

    # Prompt user for OpenAI API key
    api_key = st.text_input("Enter your OpenAI API key:")

    if not api_key:
        st.warning("Please enter your OpenAI API key before uploading the PDF file and asking a question.")
        return

    # Initialize text variable
    text = ""

    # Upload PDF files
    pdfs = st.file_uploader("Upload your books (pdf)", type="pdf", accept_multiple_files=True)

    # Extract text from PDF files
    if pdfs is not None:
        all_text = ""
        if isinstance(pdfs, list):
            # Handle multiple files
            for pdf in pdfs:
                pdf_reader = PdfReader(pdf)
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                all_text += text + "\n"
        else:
            # Handle single file
            pdf_reader = PdfReader(pdfs)
            all_text = "\n".join([page.extract_text() for page in pdf_reader.pages])

        # Split text into chunks for processing
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(all_text)

        # Check if there are any chunks to process
        if not chunks:
            st.warning("Please upload a PDF file or multiple PDF files to continue.")
            return

        # Create embeddings for the text chunks
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Ask user for question to answer
        user_question = st.text_input("What would you like to know?")

        # Answer the user's question using the language model
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(openai_api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)

            # Display the response on the Streamlit page
            st.write(response)

if __name__ == '__main__':
    main()
