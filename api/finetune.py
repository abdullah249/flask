import os
from flask import Flask, request
import pinecone
import openai
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone

# Set the OPENAI_API_KEY environment variable
os.environ['OPENAI_API_KEY'] = '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'

# Initialize the Flask app
app = Flask(__name__)

# Define the route for '/chat/'
@app.route('/chat/', methods=['GET', 'POST'])
def home():
    # Configure OpenAI API key
    api = "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"
    openai.api_key = api

    if request.method == 'GET':
        query = request.args.get("question")
        answer = get_answer(query)
        return answer

# Function to load documents from a directory
def load_docs(directory):
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to get similar documents using Pinecone
def get_similar_docs(query, k=2, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Function to get an answer using the question-answering chain
def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

# Initialize Pinecone
pinecone.init(
    api_key="7c92cd5b-74c0-4bdd-a698-fe689ca3fbc2",
    environment="us-west1-gcp-free"
)

# Create Pinecone index and load documents
index_name = "pine"
documents = load_docs('Finetune')
docs = split_docs(documents)
index = Pinecone.from_documents(docs, OpenAIEmbeddings(chunk_size=1), index_name=index_name)

# Load the question-answering chain
model_name = "gpt-3.5-turbo"
llm = OpenAI(model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")

# Driver function
if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
