# Medical Reservation Chatbot

This project is a medical reservation system that utilizes a language model to assist users in booking medical appointments through a conversational interface.

## Features

- Question-answering system based on custom documents using embeddings
- Fast retrieval with FAISS vector database
- Retrieval-Augmented Generation (RAG) for dynamic responses
- Web interface using Streamlit
- Backend API built with FastAPI
- Modular architecture for scalability and integration

## Tech Stack

- Python
- FAISS
- Hugging Face Transformers
- LangChain
- FastAPI
- Streamlit

## Folder Structure
medical_reservation_chat/
│
├── app.py # Main Streamlit app
├── medical_reservation_chat.ipynb # Notebook for experimentation and prototyping
├── data/ # Custom medical documents
├── vector_store/ # FAISS index files
├── backend/ # FastAPI backend files (if separated)
└── utils/ # Helper functions and processing scripts


## Setup Instructions

1. Clone the repository:
2. Navigate to the project directory:
3. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
4. Install dependencies:
pip install -r requirements.txt
5. Run the Streamlit app:
streamlit run app.py

## Notes

- Make sure to set your Hugging Face token or API keys via environment variables or `.env` file if required.
- You can customize the documents under the `data/` directory to fit your use case.
- Backend APIs (if used) should be started separately using `uvicorn`.

## License

This project is licensed under the MIT License.




