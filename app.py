# app.py

import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# --- Part 1: Setup ---
# This part runs only once when the Space starts up.
try:
    # Securely load the API key from Space secrets
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.2)

    # The prompt template for the RAG chain
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context: {context}
        Question: {question}
        """
    )
    app_ready = True
except Exception as e:
    app_ready = False
    app_init_error = e

# This global variable will hold our RAG chain after a video is loaded
rag_chain = None

# --- Part 2: Core Logic Functions ---

# This function loads a video, creates a vector store, and builds the RAG chain
def load_youtube_video(youtube_url: str) -> str:
    global rag_chain
    if not app_ready:
        return f"Application failed to initialize. Error: {app_init_error}"
    try:
        # Extract video ID from URL
        if "watch?v=" in youtube_url:
            video_id = youtube_url.split("watch?v=")[1].split("&")[0]
        else:
            return "Error: Invalid YouTube URL format. Please use a full 'watch?v=' URL."

        # This is the most robust method for getting the transcript
        transcript_object = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(['en'])
        transcript_chunks = transcript_object.fetch()
        transcript = " ".join(chunk["text"] for chunk in transcript_chunks)
        
        if not transcript:
             return "Error: Could not fetch transcript content."

        # Split transcript into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        # Create a FAISS vector store from the transcript chunks
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Create a retriever from the vector store
        retriever = vector_store.as_retriever()

        # Define the RAG chain that will be used to answer questions
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return f"Successfully loaded transcript for video ID: {video_id}. You can now ask questions."
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except Exception as e:
        return f"An error occurred: {e}"

# This function answers a question using the globally stored RAG chain
def answer_question(question: str, history: list) -> str:
    if not rag_chain:
        return "Please load a YouTube video first before asking a question."
    try:
        # Invoke the RAG chain with the user's question
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"An error occurred while answering the question: {e}"

# --- Part 3: The User Interface ---
# We use Gradio Blocks for a custom layout
with gr.Blocks() as iface:
    gr.Markdown("# YouTube Video Q&A with RAG")
    gr.Markdown("First, enter a YouTube video URL and click 'Load Video'. This will create a knowledge base from the video's transcript. Then, ask questions about the video content in the chatbot below.")

    with gr.Row():
        url_input = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        load_button = gr.Button("Load Video")
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.ChatInterface(
        fn=answer_question,
        title="Video Chatbot",
        examples=[["What is DeepMind?"], ["Can you summarize the video?"]]
    )

    # Link the button click to the load_youtube_video function
    load_button.click(
        fn=load_youtube_video,
        inputs=[url_input],
        outputs=[status_output]
    )

# This starts the app
iface.launch(server_name="0.0.0.0", server_port=10000)
