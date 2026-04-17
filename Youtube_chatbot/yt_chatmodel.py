import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥")
st.title("YouTube Chatbot")

# ----------------------------
# API key / model
# ----------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# ----------------------------
# User input
# ----------------------------
mode = st.radio("Choose input method", ["YouTube Video", "Paste Transcript"])

if mode == "YouTube Video":
    video_id = st.text_input("Enter Video ID (e.g., kEtGm75uBes):")
else:
    manual_transcript = st.text_area("Paste transcript here (recommended if auto-fetch fails)")

question = st.text_input("Ask a question based on the video:")

# ----------------------------
# Helpers
# ----------------------------
def clean_transcript(text: str) -> str:
    """Clean noisy transcript text."""
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    text = re.sub(
        r"\b\d+\s+minutes?,?\s+\d+\s+seconds?\b",
        " ",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_transcript(video_id: str) -> str:
    """Fetch transcript in Hindi first, then English."""
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["hi", "en"])
    return " ".join(snippet.text for snippet in transcript.snippets)


def build_vectorstore(full_text: str):
    """Create vector store from transcript."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.create_documents([full_text])

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(docs, embedder)
    return vectorstore


def get_relevant_context(full_text: str, query: str, k: int = 5) -> str:
    """Retrieve the most relevant chunks for a question."""
    vectorstore = build_vectorstore(full_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in relevant_docs)

def detect_language(text):
    return "hi" if any('\u0900' <= c <= '\u097F' for c in text) else "en"

def ask_groq(context: str, question: str) -> str:
        
    lang = detect_language(question)

    if lang == "en":
        language_instruction = "Answer ONLY in English."
    else:
        language_instruction = "Answer ONLY in Hindi."

    
    prompt = f"""
    ""Generate answer using Groq.""
You are a helpful assistant.

{language_instruction}

Use only the context below to answer the user's question.
If the user asks for a summary, provide a clear and concise summary.
If the answer is not present in the context, say:
"I could not find that in the transcript."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": ("Answer only from the provided transcript context."
                            "You must reply in the same language as the user's question. "
                            "If the question is in English, reply only in English. "
                            "If the question is in Hindi, reply only in Hindi. "
                            "Do not copy the language of the context unless it matches the user's question.")
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content

def summarize_long_text(text: str) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    docs = splitter.create_documents([text])
    docs = docs[:8]
    
    partial_summaries = []

    # Step 1: summarize each chunk
    for doc in docs:
        chunk = doc.page_content

        prompt = f"""
Summarize the following part of a transcript clearly:

{chunk}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You summarize transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )

        summary = response.choices[0].message.content
        partial_summaries.append(summary)

    # Step 2: combine summaries
    combined_text = "\n".join(partial_summaries)

    # Step 3: final summary
    final_prompt = f"""
Create a final structured summary from the following summaries:

{combined_text}

Make it clear, concise, and well-organized.
"""

    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You create structured summaries."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )

    return final_response.choices[0].message.content

def is_summary_query(question: str) -> bool:
    q = question.lower().strip()
    summary_keywords = [
        "summary",
        "summarize",
        "summarise",
        "brief",
        "overview",
        "gist"
    ]
    return any(word in q for word in summary_keywords)


# ----------------------------
# Main flow
# ----------------------------

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        transcript_text = ""

        # Step 1: get transcript
        try:
            with st.spinner("Fetching transcript..."):
                if mode == "Paste Transcript":
                    if not manual_transcript.strip():
                        st.warning("Please paste the transcript.")
                        st.stop()
                    transcript_text = manual_transcript.strip()

                elif mode == "YouTube Video":
                    if not video_id.strip():
                        st.warning("Please enter a video ID.")
                        st.stop()
                    transcript_text = fetch_transcript(video_id.strip())

        except Exception as e:
            st.error(
                "Could not fetch transcript automatically.\n\n"
                "This often happens on Streamlit Cloud because YouTube blocks cloud IPs.\n\n"
                "What you can do:\n"
                "1. Switch to 'Paste Transcript'\n"
                "2. Paste the transcript manually\n"
                "3. Or run the app locally\n\n"
                f"Technical error: {e}"
            )
            st.info("Tip: Paste Transcript mode is the most reliable on cloud deployments.")
            st.stop()

        # Step 2: generate answer only if transcript fetch succeeded
        try:
            with st.spinner("Generating answer..."):
                transcript_text = clean_transcript(transcript_text)

                if is_summary_query(question):
                    answer = summarize_long_text(transcript_text)
                else:
                    context = get_relevant_context(transcript_text, question, k=5)
                    answer = ask_groq(context, question)
        
                st.subheader("Response:")
                st.write(answer)

        except Exception as e:
            st.error(
                    "Error generating answer.\n\n"
                    "Possible reasons:\n"
                    "- Groq API issue\n"
                    "- Transcript/context too large\n"
                    "- Embedding or vector search failure\n\n"
                    f"Technical error: {e}"
                )