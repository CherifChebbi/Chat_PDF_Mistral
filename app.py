import streamlit as st
import ollama
import PyPDF2
import time
import langdetect
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Cache pour éviter de recalculer les embeddings
@st.cache_resource()
def load_embeddings(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    texts = text_splitter.split_text(pdf_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(texts, embeddings)
    return vector_db

# Extraction rapide du texte des PDF
def extract_text_from_pdfs(uploaded_files):
    text = ""
    for pdf in uploaded_files:
        reader = PyPDF2.PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Détection de la langue de la question
def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return "fr" if lang == "fr" else "en"
    except:
        return "en"  # Par défaut en anglais si détection échoue

# Fonction d'interrogation Mistral avec réponse dans la bonne langue
def query_mistral(query, retriever):
    docs = retriever.get_relevant_documents(query, k=3)  # Limite à 3 documents pour éviter la surcharge
    context = "\n".join([doc.page_content[:1000] for doc in docs])  # Tronquer à 1000 caractères max par document
    lang = detect_language(query)

    prompt = f"""
    You are a PDF assistant. Answer the question based only on the following document context:
    \n{context}\n
    Question: {query}
    """
    if lang == "fr":
        prompt = f"""
        Vous êtes un assistant PDF. Répondez à la question uniquement en fonction du contexte du document suivant :
        \n{context}\n
        Question : {query}
        """

    start_time = time.time()
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    
    return response["message"]["content"], round(end_time - start_time, 2)

# Interface utilisateur améliorée
st.set_page_config(page_title="Chat PDF AI", layout="wide")
st.title("📄 Chat with Your PDF using Mistral AI")
st.sidebar.header("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader("Upload your PDF(s)", accept_multiple_files=True, type=["pdf"])

# Historique des conversations avec icônes
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdfs(uploaded_files)
        vector_db = load_embeddings(pdf_text)
        retriever = vector_db.as_retriever()

    # Interface de chat
    query = st.text_input("💬 Ask a question based on the document:")

    if query:
        with st.spinner("Fetching answer..."):
            response, response_time = query_mistral(query, retriever)# Affichage du chat (style ChatGPT)
            
            # Ajouter à l'historique
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("ai", response))

    # Affichage du chat (style ChatGPT)
    for role, text in reversed(st.session_state.chat_history):  # Inverser l'ordre des messages
        if role == "user":
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-end;">
                <div style="background-color: #e2f7e0; padding: 10px; border-radius: 10px; max-width: 70%; color: #333; font-family: 'Arial', sans-serif; font-size: 14px;">
                    <b>User:</b> {text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <img src="https://cdn-icons-png.flaticon.com/512/4712/4712039.png" width="40" height="40" style="margin-right: 10px; border-radius: 50%; border: 2px solid #007BFF;">
                <div style="background-color: #d1e7ff; padding: 10px; border-radius: 10px; max-width: 70%; color: #333; font-family: 'Arial', sans-serif; font-size: 14px;">
                    <b>AI:</b> {text}
                </div>
            </div>
            """, unsafe_allow_html=True)
