
# 🧠 RAG-Powered Chatbot using PDFs and Websites

This project implements a Retrieval-Augmented Generation (RAG) chatbot using OpenAI's GPT models. It allows you to chat with knowledge extracted from PDFs and websites by converting them into a vector database.

---

## 📁 Project Structure

```
.
├── pdfs/               # Folder to store source PDF documents
├── vector_db/          # Folder where vector database files are saved
├── .env                # Environment file containing your OpenAI API key
├── chat_history.json   # JSON file storing previous user conversations
├── pdf_vector.py       # Script to convert PDFs to vector embeddings
├── web_vector.py       # Script to extract and convert website content to vectors
├── chatbot.py          # Main chatbot interface
├── requirements.txt    # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/subinvasanthan/ragchabot.git
cd your-repo-name
```

### 2. Install Dependencies

Use `pip` to install the required libraries:

```bash
pip install -r requirements.txt
```

---

### 3. Set Up Environment Variables

Create a `.env` file in the root directory with the following content:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

> 🔐 Replace `your_openai_api_key_here` with your actual OpenAI API key.

---

### 4. Add PDF Documents

Place all your source PDF files inside the `pdfs/` folder.

---

### 5. Generate Vector Database from PDFs

Run the following command to create embeddings and save the vector database:

```bash
python pdf_vector.py
```

---

### 6. Generate Vector Database from Websites (Optional)

You can also include website content by editing and running:

```bash
python web_vector.py
```

This script scrapes website content and stores it into the same `vector_db/` folder.

---

### 7. Run the Chatbot

Once the vector database is ready, start the chatbot:

```bash
streamlit run chatbot.py 
```

The chatbot will:
- Use OpenAI's GPT model for chat completion
- Retrieve context from your vector DB (`vector_db/`)
- Use `chat_history.json` to maintain session memory

---

## 🧠 How It Works

1. PDFs and/or websites are processed into text chunks.
2. These chunks are embedded using OpenAI embeddings (or another model) and saved into a local vector database.
3. The chatbot script loads the vector DB and retrieves relevant chunks for each user query.
4. These chunks are passed as context to GPT for more informed, grounded answers.

---

## 📝 Notes

- Chat history is saved in `chat_history.json`. You can reset this by clearing the file contents.
- Only use content from your own documents and websites unless you have permission to use external sources.
- If you're deploying or scaling this, consider switching to a more robust vector DB like Pinecone, Weaviate, or Qdrant.

---

## 📬 Questions?

Feel free to open an issue or fork the project for your own customization.
