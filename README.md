# langchain-QA-pdf
This is a Python script that uses the Streamlit library to create a web application for asking questions about a PDF book. Here's what the code does:

1. Imports necessary libraries: streamlit for creating the web app, PyPDF2 for reading PDF files, and several other libraries from the langchain package for natural language processing tasks such as text splitting, embeddings, vector stores, question answering, and more.

2. Defines a main() function that sets up the Streamlit page configuration, displays a header and prompts the user to enter their OpenAI API key.

3. If the user enters an API key, they can upload a PDF file using the file_uploader() function. The uploaded PDF file is then read using PyPDF2 and its text content is extracted and split into chunks of 1000 characters with 200 characters overlap between each chunk. These chunks are used to create embeddings using the OpenAIEmbeddings class.

4. The user is then prompted to enter a question related to the uploaded PDF file. When the user enters a question, the script performs similarity search on the chunks using the embeddings created earlier to find the most relevant documents. The Question Answering chain is then loaded and applied to the relevant documents to provide an answer to the user's question.

5. Finally, the response from the Question Answering chain is displayed on the Streamlit page using the st.write() function.

5. The if __name__ == '__main__': block ensures that the main() function is only executed when the script is run directly (i.e., not imported as a module).


# Installation

To install the repository, please clone this repository and install the requirements:

pip install -r requirements.txt

# Usage

To use the application, run the main.py file with the streamlit CLI (after having installed streamlit):

streamlit run app.py
Contributing

This repository is for educational purposes only and is not intended to receive further contributions. It is supposed to be used as support material for the YouTube tutorial that shows how to build the project.
