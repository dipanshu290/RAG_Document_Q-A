# 📚 Research Paper Q&A Chatbot


A **Streamlit-based RAG (Retrieval-Augmented Generation) Chatbot** that allows users to upload research papers (PDF) and ask questions based on their content. Powered by **Groq LLM** and **OpenAI embeddings**, it supports **English** and **Hindi**, voice input, and text-to-speech (TTS) responses.

---

🎥 Check out the **video preview** and announcement post on LinkedIn:  
[🚀 Research Paper Q&A Chatbot – LinkedIn Demo](https://www.linkedin.com/feed/update/urn:li:activity:7365394386513301504/)

This post includes a walkthrough of the app in action, key features, and how it empowers research workflows using AI.

---

## 🌟 Features

- Upload multiple **PDF research papers** for analysis.
- Ask questions and get answers based on **document content**.
- **Voice input** support for hands-free queries.
- **Text-to-speech** output for answers.
- **Semantic search** using OpenAI embeddings and FAISS vector store.
- **Groq LLM** integration for fast, accurate responses.
- Maintain **chat history** with timestamps and ability to clear it.
- **Document similarity view** to check referenced content with FAISS.

---

## ⚠️ Notes

- This project uses **OpenAI embeddings**, so an **OpenAI API key is mandatory**.
- OpenAI usage may incur a **minimum cost of $5 USD**, which typically amounts to **₹500–₹550 INR** after forex charges.If you are comfortable to pay that much amount then head to (https://platform.openai.com/docs/api-reference/introduction) and signup
- 💡 If you prefer a free alternative, you can switch to Ollama or Hugging Face embedding models. These options work well for local or open-source setups and can be integrated with FAISS or other vector stores.
- Ensure your **microphone** is working for voice input.
- Only **PDF files** are supported.
- Groq API key must be valid and start with `gsk_`.
- Embedding creation may take a few seconds depending on PDF size.
- Voice input uses Google Speech Recognition — internet connection required.
- TTS playback uses `gTTS` and may vary slightly in pronunciation.
- App styling includes custom gradients and hover effects for better UX.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create & Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

---

## 🚀 Running the Streamlit App

1. Start the app:
   ```bash
   streamlit run app.py
   ```

2. Enter your **Groq API Key** and **OpenAI API Key** in the sidebar.
   * Groq key must start with `gsk_`.

3. Select your **language** (English or Hindi).

4. Upload one or more **PDF research papers**.

5. Enter your **query** (or use voice input).

6. Click **Search** to generate an answer.
   * The response will appear in a styled card.
   * TTS playback will automatically play the answer.

7. Explore **document similarity** for context reference.

8. Check **chat history** in the sidebar and clear history if needed.

---

## 🎛️ Optional Settings

- **Response Creativity:** Adjust via the temperature slider (0–1).
- **Voice Input:** Click the microphone button and speak your query.
- **TTS Output:** Automatically plays answers in English or Hindi.

---

## 🧩 Code Overview

1. **Environment Setup:** Loads `.env` and API keys.
2. **Sidebar:** Groq API key input, OpenAI key input, language toggle, chat history.
3. **Main UI:** Title, subtitle, file uploader with gradient styling.
4. **Vector Embeddings:** Uses OpenAI embeddings & FAISS for semantic search.
5. **Groq LLM:** Generates answers based on context.
6. **Voice Input:** Uses `speech_recognition` for transcription.
7. **Text-to-Speech:** Uses `gTTS` for audio playback.
8. **Chat History:** Stores questions, answers, and timestamps.
9. **Document Similarity:** Displays referenced document content.

---

## 📄 Example Usage

- Upload `sample_paper.pdf`.
- Ask: `"What are the key findings?"`
- See **response card**, **audio playback**, and **similarity docs**.

---

## 📝 Credits

- Developed by **Biswojit Bal**
- Powered by **Groq** & **OpenAI**
- Built with **Streamlit**, **LangChain**, **FAISS**, and **Python**
