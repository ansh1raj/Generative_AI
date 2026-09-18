import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.title("📄 PDF Question Answering")

# --------------------------------
# 1. Upload PDF
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --------------------------------
    # 2. Load PDF
    # --------------------------------

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    st.write(f"PDF loaded: {len(documents)} pages")

    # --------------------------------
    # 3. Split documents into chunks
    # --------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    st.write(f"Created {len(chunks)} chunks")

    # --------------------------------
    # 4. Create embedding model
    # --------------------------------

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------
    # 5. Create vector database
    # --------------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # --------------------------------
    # 6. Create retriever
    # --------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # --------------------------------
    # 7. Create LLM
    # --------------------------------

    llm = ChatMistralAI(
        model="ministral-14b-2512"
    )

    # --------------------------------
    # 8. Create prompt
    # --------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.success("PDF is ready! You can now ask questions.")

    # --------------------------------
    # 9. Ask question
    # --------------------------------

    query = st.text_input("Ask a question about the PDF:")

    if query:

        # Retrieve relevant documents
        docs = retriever.invoke(query)

        # Create context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Create final prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        # Get answer from LLM
        response = llm.invoke(final_prompt)

        st.write("### 🤖 AI Answer")
        st.write(response.content)