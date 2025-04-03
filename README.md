# Angel One Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot trained on Angel One's customer support documentation to assist users by answering queries and providing relevant support information.

## Features

- Answers questions based on Angel One's support documentation
- Web scraping from angelone.in/support
- PDF document processing
- Modern web interface
- Real-time chat functionality
- "I don't know" responses for unknown queries
- Uses HuggingFace models for embeddings and text generation

## Prerequisites

- Python 3.9+
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data/pdfs
mkdir -p data/chroma
```

5. Place your insurance PDFs in the `data/pdfs` directory.

## Usage

1. Start the server:
```bash
uvicorn app.main:app --reload
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Start chatting with the bot!

## API Endpoints

- `POST /api/initialize`: Initialize the chatbot with documents
- `POST /api/chat`: Send a message to the chatbot

## Project Structure

```
rag_chatbot/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── rag/
│   │   ├── chat.py
│   │   └── document_processor.py
│   ├── static/
│   ├── templates/
│   │   └── index.html
│   └── main.py
├── data/
│   ├── pdfs/
│   └── chroma/
├── requirements.txt
└── README.md
```

## Technical Details

- Uses HuggingFace's `all-MiniLM-L6-v2` model for embeddings
- Uses HuggingFace's `flan-t5-large` model for text generation
- Implements document deduplication to avoid redundant information
- Optimized prompt template for better response quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 