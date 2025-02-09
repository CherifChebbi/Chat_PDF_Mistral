# Chat PDF AI with Mistral

This project allows users to upload PDF files and interact with them using AI-powered chat through the **Mistral** model. The application leverages multiple technologies including **Streamlit**, **Ollama** for AI interaction, **LangChain** for text embeddings, and **PyPDF2** for extracting text from PDFs.

### Features:
- Upload multiple PDFs to extract and analyze the text.
- Ask questions based on the content of the uploaded PDFs.
- The AI responds based on the context of the document using the **Mistral AI** model.
- Multi-language support (English and French).
- Displays conversation history with the user and AI responses.

---

## Technologies Used:
- **Streamlit**: Framework for creating interactive web applications.
- **Ollama**: AI model used for providing intelligent responses based on the document's content.
- **PyPDF2**: Library for extracting text from PDFs.
- **LangChain**: Used for creating embeddings of the extracted PDF text and querying with vector databases (FAISS).
- **LangDetect**: Language detection for supporting both French and English languages.

---
**Install the required dependencies:**
pip install -r requirements.txt
---
