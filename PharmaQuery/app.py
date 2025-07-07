import os
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

@st.cache_resource
def get_db(api_key):
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )

    db = Chroma(
        collection_name="pharma_database",
        embedding_function=embedding_model,
        persist_directory='./pharma_db'
    )
    return db

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def add_to_db(uploaded_files, api_key):
    if not uploaded_files:
        st.error("No files uploaded!")
        return

    db = get_db(api_key)

    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_size=256, 
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(data)
        
        db.add_documents(chunks)
        os.remove(temp_file_path)

def run_rag_chain(query, api_key):
    db = get_db(api_key)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    PROMPT_TEMPLATE = """
    You are a highly knowledgeable assistant specializing in pharmaceutical sciences.
    Answer the question based only on the following context:
    {context}

    Answer the question based on the above context: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_key=api_key,
        temperature=0.7 
    )

    output_parser = StrOutputParser()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | chat_model
        | output_parser
    )
    
    response = rag_chain.invoke(query)
    return response

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="PharmaQuery", page_icon=":microscope:")
    st.header("Pharmaceutical Insight Retrieval System")

    with st.sidebar:
        st.title("Configuration")
        
        api_key_input = st.text_input(
            "Enter your Google Gemini API key:", 
            type="password",
            help="Get your key from https://aistudio.google.com/app/apikey"
        )
        
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
            st.success("API key accepted!", icon="🔑")
        
        st.markdown("---")

        if 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key:
            st.subheader("Add to Knowledge Base")
            pdf_docs = st.file_uploader(
                "Upload your research documents (PDFs):",
                type=["pdf"],
                accept_multiple_files=True
            )
            
            if st.button("Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing your documents..."):
                        add_to_db(pdf_docs, st.session_state.gemini_api_key)
                        st.success("Documents successfully added to the database!")
                else:
                    st.warning("Please upload at least one PDF file.")
        else:
            st.info("Please enter your API key to enable document uploads.")

    st.info("Ask any question about the documents you've uploaded.")
    
    if 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key:
        query = st.text_area(
            "Enter your query:",
            placeholder="e.g., What are the AI applications in drug discovery?"
        )

        if st.button("Submit Query"):
            if not query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    result = run_rag_chain(query, st.session_state.gemini_api_key)
                    st.write(result)
    else:
        st.warning("Please enter your API key in the sidebar to begin.")

    st.sidebar.markdown("---")
    st.sidebar.write("Built with by Dev")

if __name__ == "__main__":
    main()