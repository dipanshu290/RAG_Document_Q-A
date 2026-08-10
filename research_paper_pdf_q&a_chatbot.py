import streamlit as st
import os
import time
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import speech_recognition as sr
from gtts import gTTS
import base64
from io import BytesIO

# 🌱 Load environment variables
load_dotenv()

# 🎨 Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# ------------------------------
# 🌟 Sidebar UI & API Keys
# ------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(to right, #FF512F, #DD2476);
    color: white;
}
[data-testid="stSidebar"] .stTextInput>div>div>input {
    background-color: rgba(255,255,255,0.95);
    color: black;
    border-radius: 8px;
    padding: 8px;
}
.stButton>button {
    background: linear-gradient(to right, #FF512F, #DD2476);
    color: white;          /* normal text color */
    font-weight: bold;
    border-radius: 10px;
    padding: 12px 24px;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(to right, #FF512F, #DD2476);  /* keep same gradient */
    color: black;          /* text color on hover */
}

</style>
""", unsafe_allow_html=True)

# 🟢 Groq API Key
groq_api_key = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password")
if groq_api_key and not groq_api_key.startswith("gsk_"):
    st.sidebar.warning("❌ Invalid Groq API Key. Must start with 'gsk_'.")
    groq_api_key = None

# 🟢 OpenAI API Key
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
if openai_api_key and not openai_api_key.startswith("sk-"):
    st.sidebar.warning("❌ Invalid OpenAI API Key. Must start with 'sk-'.")
    openai_api_key = None

# ✅ Check both keys before proceeding
if not groq_api_key or not openai_api_key:
    st.info("Please enter **both** Groq and OpenAI API Keys to proceed.")
    st.stop()
else:
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

# ------------------------------
# 🌍 Language Toggle
# ------------------------------
language = st.sidebar.selectbox("🌐 Choose Language", ["English", "हिन्दी"])

TEXT = {
    "English": {
        "title": "📚 Research Paper Q&A Chatbot",
        "subtitle": "Ask questions based on your uploaded research papers.",
        "upload": "📁 Upload your research papers (PDF only)",
        "query": "🔍 Enter your query from the research papers:",
        "search": "🚀 Search",
        "response": "✅ Response",
        "similarity": "📄 Document similarity search",
        "footer": "Made with ❤️ by Biswojit Bal | Powered by Groq & OpenAI",
        "upload_warning": "⚠️ Please upload at least one PDF to begin.",
        "query_warning": "⚠️ Please enter a query before searching.",
        "embedding_success": "✅ Vector embeddings created successfully.",
        "embedding_error": "❌ Error creating embeddings:",
        "retrieval_error": "❌ Error during retrieval or generation:",
        "model_error": "❌ Failed to initialize Groq model:",
        "clear_history": "🧹 Clear Chat History",
        "speak": "🔊 Speak Response"
    },
    "हिन्दी": {
        "title": "📚 शोध पत्र प्रश्नोत्तर चैटबॉट",
        "subtitle": "अपने अपलोड किए गए शोध पत्रों के आधार पर प्रश्न पूछें।",
        "upload": "📁 अपने शोध पत्र अपलोड करें (केवल PDF)",
        "query": "🔍 शोध पत्रों से अपना प्रश्न दर्ज करें:",
        "search": "🚀 खोजें",
        "response": "✅ उत्तर",
        "similarity": "📄 दस्तावेज़ समानता खोज",
        "footer": "❤️ से बनाया गया Biswojit Bal द्वारा | Groq और OpenAI द्वारा संचालित",
        "upload_warning": "⚠️ कृपया शुरू करने के लिए कम से कम एक PDF अपलोड करें।",
        "query_warning": "⚠️ कृपया खोजने से पहले एक प्रश्न दर्ज करें।",
        "embedding_success": "✅ वेक्टर एम्बेडिंग सफलतापूर्वक बनाई गई।",
        "embedding_error": "❌ एम्बेडिंग बनाते समय त्रुटि:",
        "retrieval_error": "❌ पुनः प्राप्ति या उत्तर निर्माण में त्रुटि:",
        "model_error": "❌ Groq मॉडल प्रारंभ करने में विफल:",
        "clear_history": "🧹 चैट इतिहास साफ़ करें",
        "speak": "🔊 उत्तर सुनें"
    }
}

# ------------------------------
# 🎨 Main UI Styling
# ------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #FDEB71, #F8D800);
    font-family: 'Poppins', sans-serif;
}
h1 {
    background: -webkit-linear-gradient(#FF512F, #DD2476);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 42px;
}
h2 {
    color: #DD2476;
}
.response-card {
    background: linear-gradient(to right, #FF512F, #DD2476);
    color: white;
    padding: 15px;
    border-radius: 12px;
    font-size: 16px;
}

/* Drag & Drop area gradient (keep text normal) */
.stFileUploader > div {
    background: linear-gradient(to right, #36D1DC, #5B86E5);
    padding: 12px;
    border-radius: 12px;
    transition: all 0.3s ease;
}

/* Hover effect for the box */
.stFileUploader > div:hover {
    background: linear-gradient(to right, #5B86E5, #36D1DC);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* 🔹 Text Area Gradient */
.stTextArea>div>div>textarea {
    background: linear-gradient(to right, #FF9A9E, #FAD0C4);
    color: white;
    border-radius: 12px;
    padding: 12px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stTextArea>div>div>textarea:hover,
.stTextArea>div>div>textarea:focus {
    background: linear-gradient(to right, #FAD0C4, #FF9A9E);
    color: black;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.stTextArea>div>div>textarea::placeholder {
    color: #f1f1f1;
    opacity: 0.9;
}

/* 🔹 Footer Gradient matching button */
.footer {
    text-align: center;
    padding: 12px;
    color: white;
    background: linear-gradient(to right, #FF512F, #DD2476);
    border-radius: 12px;
    margin-top: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Title & Subtitle
# ------------------------------
st.markdown(f"""
<div style="
    background: linear-gradient(to right, #FF512F, #DD2476);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
">
    <h1 style='margin:0; font-size:42px; font-weight:bold;
               background: linear-gradient(to right, #a1c4fd, #c2e9fb);
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;
               '>{TEXT[language]['title']}</h1>
    <p style='margin:5px 0 0 0; font-size:18px; font-style:italic;
              background: linear-gradient(to right, #a1c4fd, #c2e9fb);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              '>{TEXT[language]['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

temperature = st.slider("🎛️ Response creativity", 0.0, 1.0, 0.7)

# 🔐 Groq LLM Init
try:
    llm = ChatGroq(model="llama3-8b-8192", temperature=temperature)
except Exception as e:
    st.error(f"{TEXT[language]['model_error']} {e}")
    st.stop()

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided documents.
Answer the question to the best of your ability based on the context provided in the documents.
If you do not know the answer, say "I don't know".

<context>{context}</context>
Question: {input}
""")

uploaded_files = st.file_uploader(TEXT[language]["upload"], type=["pdf"], accept_multiple_files=True)

# ------------------------------
# Vector Embeddings
# ------------------------------
def create_vector_embeddings():
    try:
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(documents)
        st.session_state.vector_store = FAISS.from_documents(final_docs, st.session_state.embeddings)
        st.success(TEXT[language]["embedding_success"])
    except Exception as e:
        st.error(f"{TEXT[language]['embedding_error']} {e}")
        st.stop()

# ------------------------------
# Session State
# ------------------------------
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

# ------------------------------
# User Input
# ------------------------------
user_prompt = st.text_area(TEXT[language]["query"], value=st.session_state.user_prompt, height=100)

# ------------------------------
# 🎙️ Voice Input
# ------------------------------
st.markdown("### 🎤 Voice Input (Optional)")
if st.button("🎙️ Speak Your Question"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎧 Listening... Please speak clearly.")
        audio = recognizer.listen(source)

    try:
        voice_query = recognizer.recognize_google(audio)
        st.session_state.user_prompt = voice_query
        st.success(f"📝 Transcribed: {voice_query}")
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition error: {e}")

# ------------------------------
# 🔍 Search & TTS
# ------------------------------
if st.button(TEXT[language]["search"]):
    if uploaded_files:
        create_vector_embeddings()
        if st.session_state.user_prompt:
            try:
                with st.spinner("🔎 Searching and generating response..."):
                    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
                    retriever = st.session_state.vector_store.as_retriever()
                    retriever_chain = create_retrieval_chain(retriever, document_chain)

                    start = time.process_time()
                    response = retriever_chain.invoke({"input": st.session_state.user_prompt})
                    elapsed = time.process_time() - start

                    # Response Card
                    answer_text = response.get('answer') or response.get('output') or response.get('result') or '🤖 No response generated.'
                    st.markdown(f"<h2>{TEXT[language]['response']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div class='response-card'>{answer_text}</div>", unsafe_allow_html=True)
                    st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

                    st.session_state.last_answer = answer_text

                    # TTS
                    tts = gTTS(answer_text, lang="en" if language=="English" else "hi")
                    audio_bytes_io = BytesIO()
                    tts.write_to_fp(audio_bytes_io)
                    audio_bytes_io.seek(0)
                    b64 = base64.b64encode(audio_bytes_io.read()).decode()
                    st.markdown(f"<audio autoplay style='display:none;'><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>", unsafe_allow_html=True)

                    # Chat History
                    st.session_state.chat_history.append({
                        "question": st.session_state.user_prompt,
                        "answer": answer_text,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Similarity Docs
                    with st.expander(TEXT[language]["similarity"]):
                        context_docs = response.get("context", [])
                        if context_docs:
                            for i, doc in enumerate(context_docs):
                                st.markdown(f"**Document {i+1}:**")
                                st.write(doc.page_content)
                                st.markdown("---")
                        else:
                            st.info("📄 No relevant documents found. Answering from general knowledge.")
            except Exception as e:
                st.error(f"{TEXT[language]['retrieval_error']} {e}")
        else:
            st.warning(TEXT[language]["query_warning"])
    else:
        st.warning(TEXT[language]["upload_warning"])

# ------------------------------
# 🗂️ Sidebar Chat History
# ------------------------------
with st.sidebar:
    st.markdown("### 🗂️ Chat History")
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{i}: {chat['question']}"):
                st.markdown(f"🕒 {chat['timestamp']}")
                st.markdown(f"**Answer:** {chat['answer']}")
    else:
        st.info("💬 No chat history yet.")

    if st.button(TEXT[language]["clear_history"]):
        st.session_state.chat_history = []
        st.success("✅ Chat history cleared.")

# ------------------------------
# 🧾 Footer
# ------------------------------
st.markdown(f"<div class='footer'>{TEXT[language]['footer']}</div>", unsafe_allow_html=True)
