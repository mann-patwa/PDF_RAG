# 📄 Local PDF RAG with LlamaIndex & Hugging Face

This project is a complete, 100% free, and local-first **Retrieval-Augmented Generation (RAG)** pipeline.

You can add your own PDF documents and then ask questions about them using a local, open-source large language model (LLM). Your data never leaves your machine, ensuring 100% privacy.

---

### ✨ Features

- **PDF Processing**: Uses `LangChain` to load and split PDF documents into manageable chunks.
- **Vector Embeddings**: Uses `LlamaIndex` with a local Hugging Face `BAAI/bge-small-en-v1.5` model to create vector embeddings.
- **Vector Storage**: Uses `ChromaDB` as a persistent, local vector database.
- **Response Generation**: Uses a local Hugging Face `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model to generate answers based on the retrieved context.

---

## 🚀 How to Run This Project

Here is a complete, step-by-step guide from a fresh download to asking your first question.

### Part 1: One-Time Setup

You only need to do these steps the very first time you set up the project.

#### Step 1: Check Prerequisites

Make sure you have the following installed on your system:

- **Git**: For cloning the repository.
- **Python 3.10 or newer**: To run the scripts.

#### Step 2: Get the Code

Clone the repository to your local machine using Git:

```bash
git clone [https://github.com/mann-patwa/PDF_RAG.git](https://github.com/mann-patwa/PDF_RAG.git)
cd PDF_RAG
```

#### Step 3: Create a Virtual Environment

It is a strong best practice to use a virtual environment to keep your project's dependencies separate.

- **On macOS/Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **On Windows (Command Prompt):**
  ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
  ```
  You'll know it worked if you see `(.venv)` at the beginning of your terminal prompt.

#### Step 4: Install All Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

> **Note:** This will install several libraries, including `torch`, `transformers`, and `langchain`. This step might take a few minutes.

#### Step 5: Add Your PDF Documents

1.  Create a folder named `data` in the main project directory.
    ```bash
    mkdir data
    ```
2.  Find the PDF files you want to query and copy them into this new `data` folder.

---

### Part 2: Running the RAG Pipeline

This is the workflow you will follow to use the project.

#### Step 6: Ingest Your Documents (Run this first)

This script will process all the PDFs in your `data` folder, split them into chunks, create vector embeddings, and save them to a local database.

Run the ingestion script:

```bash
python ingest.py
```

> ⚠️ **IMPORTANT: What to Expect**
>
> - **Model Downloads:** The **first time** you run this, it will connect to Hugging Face to download the embedding model (`BAAI/bge-small-en-v1.5`) and the LLM (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`). This download can be **several gigabytes** and may take a while depending on your internet connection. This only happens once.
> - **Database Creation:** After processing, you will see a new folder named `chroma_db` in your project. This is your local vector database.
>
> You only need to re-run this script if you **add, remove, or change** the PDF files in your `data` folder.

#### Step 7: Query Your Documents (Run this anytime)

Once your database is created, you can run the query script to start asking questions.

Run the query script:

```bash
python query.py
```

> ⚠️ **IMPORTANT: What to Expect**
>
> - **Model Loading:** This script will load the local LLM into your computer's memory (RAM/VRAM), which may take a moment.
> - **Slow Responses:** You are running a powerful AI model on your own hardware, not a paid API. **Be patient!** Generating an answer can take anywhere from **10 to 60 seconds** (or more) depending on your computer's CPU/GPU.

The script will launch an interactive prompt. Just type your question and press Enter:

```
Setting up LlamaIndex settings for Hugging Face...
...
Index loaded successfully.

--- Ready to Query! ---
Type your question and press Enter. Type 'quit' to exit.

Query: What is the main topic of the document?

Processing query... (This may take a moment)

Response:
Based on the provided context, the main topic of the document is [Your model's answer here].

Query: quit
```

To exit the program, simply type `quit` and press Enter.

---

## 📁 Project Structure

Your final project folder should look like this:

```bash
YourRepoName/
├── .gitignore         # Ignores .venv, chroma_db, etc.
├── data/
│   └── my_document.pdf  # (You add your PDFs here)
├── ingest.py          # Script to process and store PDFs
├── query.py           # Script to ask questions
├── README.md          # This file
└── requirements.txt   # List of all Python dependencies
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
