# 🎥 YouTube ChatBot & Summarizer

This Streamlit app summarizes and chats with any YouTube video using AssemblyAI, OpenAI/Groq, and Langchain.

## 🔧 Features

- 🎧 Download YouTube audio
- 🧠 Transcribe + Summarize with AssemblyAI
- 💬 Chat with video content (LangChain + Groq/OpenAI)
- 🧹 Reset cache and folder storage
- 📊 Detect sensitive topics and categorize discussion

## 🚀 Demo

1. Paste a YouTube link
2. Get summary, topics, and sensitive content
3. Ask questions about the video

## 🛠️ Setup

```bash
git clone https://github.com/roshanj13/YOUTUBE-CHATBOT-.git
cd YOUTUBE-CHATBOT-
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
ASSEMBLYAI_API_KEY=your_assemblyai_key
GROQ_API_KEY=your_groq_key
```


## 🧠 Tech Stack

- Python
- Streamlit
- AssemblyAI
- OpenAI & Groq
- LangChain
