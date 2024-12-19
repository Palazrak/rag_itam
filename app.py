# Import langchain dependencies
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Bring in streamlit for UI dev
import streamlit as st
# Other libraries
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma

# Set Hugging Face credentials for Llama 3.2
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM

import torch

# Setup LLM credentials
access_token = os.getenv("HUGGING_FACE_TOKEN")
if not access_token:
    raise ValueError("HUGGING_FACE_TOKEN is not set in the environment variables.")

model_name = "meta-llama/Llama-3.2-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=access_token,
    # load_in_4bit=True,
    device_map="auto",  # Automatically assign layers to available devices,
    torch_dtype=torch.float16,
    load_in_8bit=True
)

# Optimize model with torch.compile
model = torch.compile(model)

# Set up text generation pipeline
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=256,  # Adjusted max length
    temperature=0.7,  # Introduced more balanced temperature
    top_k=50  # Filter for better responses
)

# Define LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Function to load documents from a folder
def load_documents_from_folder(folder_path):
    loaders = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    loaders.append(PyMuPDFLoader(file_path))
                elif file.endswith(".doc") or file.endswith(".docx"):
                    loaders.append(Docx2txtLoader(file_path))
                elif file.endswith(".txt"):
                    loaders.append(TextLoader(file_path))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return loaders

# Create and save vectorstore index
def create_and_save_index(folder_path, index_path):
    loaders = load_documents_from_folder(folder_path)
    if not loaders:
        raise ValueError("No valid files found in the specified folder.")

    vectorstore = Chroma(
        persist_directory=index_path,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    for loader in loaders:
        docs = loader.load()
        vectorstore.add_documents(documents=docs)

    vectorstore.persist()
    print(f"Index saved to {index_path}")


# Load precomputed vectorstore index
def load_index(index_path):
    return Chroma(
        persist_directory=index_path,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

# Improved RAG Chain
def get_response(question, index):
    retriever = index.as_retriever(search_kwargs={"k": 5})
    documents = retriever.get_relevant_documents(question)

    # Sort documents by similarity and condense context
    documents = sorted(documents, key=lambda x: x.metadata.get("similarity", 0), reverse=True)
    context = " ".join([doc.page_content[:250] for doc in documents[:3]])  # Limit to top 3 docs, 250 chars each
    try:
        response = llm(context + "\n\nQ: " + question, max_new_tokens=150, num_return_sequences=1)
        return response[0]["generated_text"] if isinstance(response, list) else response
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit App
def main():
    st.title("Datetos Helper")
    st.write("Upload your documents in the `knowledge_files` folder and ask questions!")


    knowledge_folder_path = "knowledge_files"
    index_path = "index"

    if not os.path.exists(index_path):
        try:
            create_and_save_index(knowledge_folder_path, index_path)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    index = load_index(index_path)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter your question here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        response = get_response(prompt, index)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()