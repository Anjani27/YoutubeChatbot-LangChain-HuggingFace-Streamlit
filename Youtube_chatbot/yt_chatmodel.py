import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from huggingface_hub import InferenceClient
import os
import sys
from dotenv import load_dotenv
#load_dotenv()
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"] # instead of load_dotenv() for Streamlit secrets management

# Setup Streamlit UI
st.set_page_config(page_title="YouTube Chatbot")
st.title("YouTube Chatbot")

video_id = st.text_input("Enter YouTube Video ID (e.g., kEtGm75uBes):")
ask = st.text_input("Ask a question based on the video:")

# Model
REPO_ID = "HuggingFaceH4/zephyr-7b-beta" #"google/flan-t5-base"  # Supported instruction-tuned model

# Setup Hugging Face client
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(REPO_ID, token=hf_token)


# Process Button
if st.button("Get Answer") and video_id and ask:
    with st.spinner("Processing transcript and retrieving answer..."):

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            text = " ".join([t["text"] for t in transcript])
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")
            st.stop()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([text])

        # Embeddings + Vector DB
        embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
        vectorstore = FAISS.from_documents(docs, embedder)

        
        def run_hf_llm(prompt: str) -> str:
            response = client.text_generation(
                prompt=prompt.text,
                max_new_tokens=300,
                temperature=0.2,
            )
            return response#.generated_text

        llm = RunnableLambda(run_hf_llm)

        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            llm=llm
        )

        prompt = PromptTemplate(
            template="""
                You are a helpful assistant. Answer the user's question using the transcript context below.
                If the question is unrelated to the video, say: "I can only answer questions related to the video."

                Context: {context}
                Question: {question}
                """,
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Build chain
        retriever_chain = RunnableLambda(lambda x: x) | retriever | RunnableLambda(format_docs)
        input_chain = RunnableParallel({'context': retriever_chain, 'question': RunnableLambda(lambda x: x)})
        full_chain = input_chain | prompt | llm | StrOutputParser()

        # Run Q&A
        response = full_chain.invoke(ask)

        st.subheader("Response:")
        st.write(response)
