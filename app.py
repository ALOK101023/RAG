# Final check to force redeployment
import gradio as gr
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi

# It's better practice to get the API key from the environment variables set in Render
# rather than hardcoding it or using a .env file on the server.
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    # This will show an error on the Gradio interface if the API key is not set.
    raise ValueError("OPENAI_API_KEY environment variable not set!")

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
video_db = None

def load_video(url):
    """
    Loads a YouTube video transcript, splits it into chunks,
    and creates a FAISS vector database.
    """
    global video_db
    try:
        # Check if the URL is valid by trying to get the transcript list
        video_id = url.split("v=")[1].split("&")[0]
        YouTubeTranscriptApi.list_transcripts(video_id) # This is the line that was failing

        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        transcript = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        video_db = FAISS.from_documents(docs, embeddings)

        return (
            f"Video loaded successfully! Ready to answer questions.",
            None, # Clear chatbot history
            gr.update(interactive=True), # Enable query input
            gr.update(interactive=True), # Enable ask button
        )
    # THIS IS THE MODIFIED ERROR MESSAGE FOR OUR TEST
    except Exception as e:
        return "TEST FAILED. THE NEW CODE IS RUNNING BUT THE LIBRARY IS STILL BROKEN.", None, gr.update(interactive=False), gr.update(interactive=False)


def query_chatbot(query, history):
    """
    Queries the vector database with a user question and gets a response.
    """
    global video_db
    if not video_db:
        return "Please load a video first.", history

    docs = video_db.similarity_search(query)

    if not docs:
        return "Could not find relevant information in the video.", history

    # Get the content from the top 4 relevant documents
    context = " ".join([doc.page_content for doc in docs[:4]])

    prompt_template = """
    You are a helpful assistant for the YouTube video.
    Answer the question based only on the following context from the video transcript:
    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(context=context, question=query)
    history.append((query, response))
    return "", history


with gr.Blocks() as iface:
    gr.Markdown("# YouTube Video Q&A with RAG")
    gr.Markdown(
        "First, enter a YouTube video URL and click 'Load Video'. "
        "This will create a knowledge base from the video's transcript. "
        "Then, ask questions about the video content in the chatbot below."
    )

    with gr.Row():
        youtube_url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=GfH5Of6ZBvo",
            interactive=True,
        )
        load_video_btn = gr.Button("Load Video")

    status_output = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Video Chatbot")
    query_input = gr.Textbox(
        label="Ask a question", interactive=False, placeholder="What is DeepMind?"
    )
    ask_btn = gr.Button("Ask", interactive=False)

    gr.Examples(
        examples=[
            "What is DeepMind?",
            "Can you summarize the video?",
        ],
        inputs=query_input
    )

    load_video_btn.click(
        fn=load_video,
        inputs=youtube_url_input,
        outputs=[status_output, chatbot, query_input, ask_btn],
    )

    ask_btn.click(
        fn=query_chatbot, inputs=[query_input, chatbot], outputs=[query_input, chatbot]
    )
    query_input.submit(
        fn=query_chatbot, inputs=[query_input, chatbot], outputs=[query_input, chatbot]
    )

# This line is correct for Render deployment
iface.launch(server_name="0.0.0.0", server_port=10000)
