Here is my web-app: http://pdf-ask.streamlit.app
# Langchain QA pdf
This is a Python script that uses the Streamlit library to create a web application for asking questions about a PDF book. Here's what the code does:

1. Imports necessary libraries: streamlit for creating the web app, PyPDF2 for reading PDF files, and several other libraries from the langchain package for natural language processing tasks such as text splitting, embeddings, vector stores, question answering, and more.

2. Defines a main() function that sets up the Streamlit page configuration, displays a header and prompts the user to enter their OpenAI API key.

3. If the user enters an API key, they can upload a PDF file using the file_uploader() function. The uploaded PDF file is then read using PyPDF2 and its text content is extracted and split into chunks of 1000 characters with 200 characters overlap between each chunk. These chunks are used to create embeddings using the OpenAIEmbeddings class.

4. The user is then prompted to enter a question related to the uploaded PDF file. When the user enters a question, the script performs similarity search on the chunks using the embeddings created earlier to find the most relevant documents. The Question Answering chain is then loaded and applied to the relevant documents to provide an answer to the user's question.

5. Finally, the response from the Question Answering chain is displayed on the Streamlit page using the st.write() function.

5. The if __name__ == '__main__': block ensures that the main() function is only executed when the script is run directly (i.e., not imported as a module).

# How it work?

This code is a Streamlit app that allows the user to upload a PDF file and ask questions about it using OpenAI's question answering model. Here's how it works:

1. The user is prompted to enter their OpenAI API key.

2. If the user enters a valid key, they can then upload a PDF file.

3. Once a PDF file has been uploaded, the text from the entire document is extracted using PyPDF2 library.

4. The extracted text is split into smaller chunks using a CharacterTextSplitter from the langchain package.

5. penAI embeddings are created for each chunk of text.

6. A FAISS index is built from these embeddings, creating a knowledge base.

7. The user is prompted to input a question about the PDF.

8. The knowledge base is searched for documents with similar content to the user's question.

9. An OpenAI language model is used to generate an answer to the user's question based on the documents found in step 8.

10. The generated answer is displayed to the user using Streamlit.

11. The process repeats from step 7 if the user inputs another question.

In summary, this app uses several third-party libraries and services to allow users to easily ask questions about the content of a PDF file, without having to manually search through its text.

# Installation

To set up this repository, you'll need to go through the following steps:

1. Get a copy of the repository by either downloading the ZIP file and extracting it or running  <code> git clone <repository_url> </code> in your terminal.

2. Navigate to the directory where the repository was downloaded, using the cd command in your terminal.

3. Install the necessary Python packages by executing the command <code> pip install -r requirements.txt </code>. This will install all dependencies listed in the requirements file.


# Usage
  
To run the application, follow these steps:

1. Make sure you have installed Streamlit on your machine.

2. Open a terminal window and navigate to the directory where the application is saved.

3. Run the command <code> streamlit run app.py </code> in the terminal. This will start the application.

4. Once the application has started, open a web browser and navigate to the URL provided by Streamlit.

5. You can now use the application to perform the desired task.

# Reference:
- https://github.com/gkamradt/langchain-tutorials
- https://github.com/alejandro-ao/langchain-ask-pdf
