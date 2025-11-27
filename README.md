# SIC-TinyLlama-NL2SQL

A powerful Natural Language to SQL (NL2SQL) system that leverages a locally running TinyLlama model, Retrieval-Augmented Generation (RAG), and a modern web interface to allow users to query databases using plain English.

## Features

- **Natural Language to SQL**: Convert English questions into valid SQL queries.
- **Local Model Execution**: Uses [TinyLlama-NL2SQL](https://huggingface.co/Kiyk0/TinyLlama-NL2SQL/tree/main) running locally for privacy and performance.
- **Schema-Aware RAG**: Retrieves relevant tables and schema context using vector embeddings (FAISS) to improve accuracy.
- **Interactive UI**: React-based frontend for easy interaction and result visualization.
- **Model Notebook**: Includes `model/tinyllama.ipynb` for model training and fine-tuning experiments.

## Tech Stack

- **Backend**: Python, FastAPI, Hugging Face Transformers, PyTorch, FAISS, MySQL Connector.
- **Frontend**: React, Tailwind CSS, Lucide React.
- **Database**: MySQL.
- **Model**: TinyLlama-1.1B (Fine-tuned for SQL).

## Prerequisites

- Python 3.8+
- Node.js & npm
- MySQL Server
- GPU (Recommended for faster inference, but works on CPU)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SIC-TinyLlama-NL2SQL
```

### 2. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Database Setup

Ensure you have a MySQL database named `elearning` running. Update the `.env` file in the `backend` directory with your database credentials.

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=elearning
```

### 4. Frontend Setup

```bash
cd frontend
npm install
```

## Usage

### Start the Backend

```bash
cd backend
source venv/bin/activate
python main.py
```

The API will be available at `http://localhost:5000`.

### Start the Frontend

```bash
cd frontend
npm start
```

The application will open at `http://localhost:3000`.

## Model Notebook

The project includes a Jupyter notebook `model/tinyllama.ipynb` which demonstrates how the model was trained or can be fine-tuned. You can run it using Jupyter Lab or Notebook.

```bash
jupyter notebook model/tinyllama.ipynb
```
