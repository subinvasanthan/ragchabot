import streamlit as st
import openai
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings

# ========== CONFIGURATION ==========
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
VECTOR_DB_DIR = "vector_db"
CHAT_HISTORY_FILE = "chat_history.json"

# ========== INITIALIZATION ==========
EMBEDDING_MODEL = OpenAIEmbeddings()

# ========== SESSION STATE ==========
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'combined_index' not in st.session_state:
    st.session_state.combined_index = None

if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []


# ========== CORE FUNCTIONS ==========
def load_all_vector_dbs():
    """Load and combine all vector databases"""
    if not os.path.exists(VECTOR_DB_DIR):
        st.error(f"Vector DB directory '{VECTOR_DB_DIR}' not found")
        return False

    db_files = [f for f in os.listdir(VECTOR_DB_DIR) if f.endswith('.index')]
    if not db_files:
        st.error("No vector databases found")
        return False

    all_chunks = []
    indices = []

    for db_file in db_files:
        db_name = db_file.replace('.index', '')
        db_path = os.path.join(VECTOR_DB_DIR, db_name)

        # Load FAISS index
        index = faiss.read_index(f"{db_path}.index")
        indices.append(index)

        # Load metadata
        with open(f"{db_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
            all_chunks.extend(metadata['chunks'])

    # Combine all indices into one
    if len(indices) > 1:
        combined_index = faiss.IndexShards(indices[0].d)
        for index in indices:
            combined_index.add_shard(index)
    else:
        combined_index = indices[0]

    st.session_state.combined_index = combined_index
    st.session_state.all_chunks = all_chunks
    return True


def search_similar_chunks(query, k=5):
    """Search across all knowledge bases"""
    if st.session_state.combined_index is None or not st.session_state.all_chunks:
        st.error("Knowledge base not initialized")
        return []

    query_embedding = EMBEDDING_MODEL.embed_query(query)
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = st.session_state.combined_index.search(query_vector, k)

    # Get unique chunks (in case same chunk appears in multiple indices)
    unique_indices = np.unique(indices[0])
    results = [st.session_state.all_chunks[i] for i in unique_indices if i < len(st.session_state.all_chunks)]
    return results[:k]  # Return top k unique results


def generate_response(user_input):
    """Generate chatbot response using RAG"""
    relevant_chunks = search_similar_chunks(user_input)
    if not relevant_chunks:
        return "I couldn't find relevant information to answer your question."

    # Prepare context with sources
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(f"Source: {chunk['source']}\nContent: {chunk['text']}")
    context = "\n\n".join(context_parts)

    # Prepare chat history
    history_parts = []
    for msg in st.session_state.chat_history[-3:]:
        history_parts.append(f"User: {msg['user']}\nAssistant: {msg['bot']}")
    history = "\n".join(history_parts)

    prompt = """You are a knowledgeable assistant that answers questions using ONLY the provided context.

Context from knowledge base:
{context}

Recent conversation history:
{history}

Current question: {user_input}

Guidelines:
1. Answer concisely using ONLY the provided context
2. If the answer isn't in the context, say "I don't have information about that"
3. Always cite sources when available
4. Be professional and helpful
""".format(context=context, history=history, user_input=user_input)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


def save_chat_history():
    """Save chat history to file"""
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(st.session_state.chat_history, f)


def load_chat_history():
    """Load chat history from file"""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


# ========== STREAMLIT UI ==========
def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 Unified Knowledge Chatbot")

    # Initialize knowledge base
    if st.session_state.combined_index is None:
        with st.spinner("Loading all knowledge bases..."):
            if not load_all_vector_dbs():
                st.stop()

    # Display KB stats
    st.sidebar.markdown("**Knowledge Base Stats:**")
    st.sidebar.markdown(f"- Total Chunks: {len(st.session_state.all_chunks)}")

    source_count = len({chunk['source'] for chunk in st.session_state.all_chunks})
    st.sidebar.markdown(f"- Unique Sources: {source_count}")

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        save_chat_history()
        st.rerun()

    # Load chat history
    if not st.session_state.chat_history:
        st.session_state.chat_history = load_chat_history()

    # Display chat
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["bot"])

    # User input
    if user_input := st.chat_input("Ask me anything..."):
        # Add user message to history
        st.session_state.chat_history.append({"user": user_input, "bot": ""})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(user_input)
                st.markdown(response)

            # Update history
            st.session_state.chat_history[-1]["bot"] = response
            save_chat_history()


if __name__ == "__main__":
    main()
