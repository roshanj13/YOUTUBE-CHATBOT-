#coding part
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate# Initialize OpenAI with your API key
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
import shutil
import librosa
import openai
import soundfile as sf
import youtube_dl
from youtube_dl.utils import DownloadError
import yt_dlp as youtube_dl
import pickle
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
#load api key lib
from dotenv import load_dotenv
import base64
from langchain.memory import ConversationSummaryMemory
from langchain.chains import (
     LLMChain, ConversationalRetrievalChain
)
from langchain_groq import ChatGroq
llm = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY




#sidebar contents

with st.sidebar:
    st.title('🦜️🔗youtube based chatbot  CHATBOT🤗')
    st.markdown('''
    ## About APP:

    to do ....................
    ''')

    st.write('💡All about videos based chatbot, created by amitheshhh')

load_dotenv()
def find_audio_files(path, extension=".mp3"):
    """Recursively find all files with extension in path."""
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))

    return audio_files
def youtube_to_mp3(youtube_url: str, output_dir: str):
    """Download the audio from a youtube video, save it to output_dir as an .mp3 file.

    Returns the filename of the saved video.
    """

    # Config
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True,
        #"ffmpeg_location":os.path.realpath(r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe") # Provide the path to ffmpeg here
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    try:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        # weird bug where youtube-dl fails on the first download, but then works on second try... hacky ugly way around it.
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename
def chunk_audio(filename, segment_length: int, output_dir):
    """Segment length is in seconds"""


    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Load audio file
    audio, sr = librosa.load(filename, sr=44100)

    # Calculate duration in seconds
    duration = librosa.get_duration(y=audio, sr=sr)

    # Calculate number of segments
    num_segments = int(duration / segment_length) + 1


    # Iterate through segments and save them
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)

    chunked_audio_files = find_audio_files(output_dir)
    return sorted(chunked_audio_files)
def transcribe_audio(audio_files: list, output_file=None, model="whisper-1"):
    
    transcripts = []
    for audio_file in audio_files:
        audio = open(audio_file, "rb")
        response = openai.Audio.transcribe(model, audio)
        transcripts.append(response["text"])

    if output_file is not None:
        # Save all transcripts to a .txt file
        with open(output_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript + "\n")

    return transcripts
def get_transcript(youtube_url: str, outputs_dir: str):
    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/chunks"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    # Download audio from youtube
    audio_filename = youtube_to_mp3(youtube_url, outputs_dir)
    segment_length = 10 * 60
    
    audio_filename =  youtube_to_mp3(youtube_url,raw_audio_dir)
    chunked_audio_files = chunk_audio(
        audio_filename, segment_length=segment_length, output_dir=chunks_dir
    )
    transcriptions = transcribe_audio(chunked_audio_files, transcripts_file)
    return transcriptions

def create_vector_store(youtube_link):
    st.write("Downloading the YouTube video...")
    transcripts = get_transcript(youtube_link, "outputs")
    doc = Document(page_content="\n".join(transcripts), metadata={"source": youtube_link})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_documents([doc])
    print(chunks)
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    db = Chroma.from_documents(chunks, embeddings, persist_directory=f"./new_db_{youtube_link.replace('https://www.youtube.com/watch?v=', '')}")
    db.persist()
    
def clear_folders():
    audio_dir = "audio"
    output_dir = "outputs"
    if os.path.exists(audio_dir):
        shutil.rmtree(audio_dir)
    os.makedirs(audio_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    if "vector_store_created" in st.session_state:
        del st.session_state.vector_store_created
        st.experimental_rerun()
def main():
   
    st.header("📄Chat with content from a YouTube video🤗")
    # Upload a YouTube link
    clear_button = st.button("Clear Folders and Reset")
    if clear_button:
        clear_folders()
    youtube_link = st.text_input("Paste your YouTube link here")
    if youtube_link:
        try:
            if "vector_store_created" not in st.session_state:
                create_vector_store(youtube_link)
                st.session_state.vector_store_created = True
                embeddings = OllamaEmbeddings(model='nomic-embed-text')
                db = "new_db_"+youtube_link.split('=')[1]
                
                
                st.session_state.retriever = Chroma(persist_directory=f"./{db}", embedding_function=embeddings).as_retriever()

            prompt_template = """You are an assistant for question-answering on a given video transcript.
            Use the following pieces of retrieved context, here context refers to the transcript of the video related to the given question use this to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.
            Question: {question}
            Context: {context}
            Answer:
            """
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": PROMPT}
            # Accept user questions/query
            query = st.text_input("Ask questions related to the content of the video")
            if query:
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    chain_type="stuff",
                    memory=ConversationSummaryMemory(llm=llm, memory_key='chat_history', input_key='question', output_key='answer', return_messages=True),
                    retriever=st.session_state.retriever,
                    return_source_documents=False,
                    combine_docs_chain_kwargs=chain_type_kwargs)

                response = chain.invoke(query)
                st.write(response['answer'])
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()