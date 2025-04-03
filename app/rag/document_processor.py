import os
from typing import List, Dict
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import time
from langchain_core.documents import Document
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size for more granular splitting
            chunk_overlap=50,  # Reduced overlap to minimize duplicates
            length_function=len,
        )
        self.vector_store = None
        self.base_url = "https://www.angelone.in/support"

    def process_documents(self, doc_dir: str) -> List[Document]:
        """Process all documents in the specified directory."""
        logger.info(f"Processing documents from directory: {doc_dir}")
        all_docs = []
        processed_content = set()  # Track unique content
        
        if not os.path.exists(doc_dir):
            logger.error(f"Document directory {doc_dir} does not exist")
            return all_docs

        for filename in os.listdir(doc_dir):
            file_path = os.path.join(doc_dir, filename)
            logger.info(f"Processing file: {filename}")
            
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif filename.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                else:
                    logger.warning(f"Skipping unsupported file: {filename}")
                    continue
                
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} pages from {filename}")
                
                # Log a sample of the content
                for i, doc in enumerate(docs[:2]):  # Log first 2 pages
                    logger.info(f"Sample content from {filename} page {i+1}:")
                    logger.info(f"{doc.page_content[:200]}...")
                
                split_docs = self.text_splitter.split_documents(docs)
                logger.info(f"Split into {len(split_docs)} chunks")
                
                # Filter out duplicate content
                for doc in split_docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in processed_content:
                        processed_content.add(content_hash)
                        all_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        logger.info(f"Total unique documents processed: {len(all_docs)}")
        return all_docs

    def get_support_links(self) -> List[str]:
        """Get all support-related links from the main support page."""
        try:
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links in the support section
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('/support') or href.startswith(self.base_url):
                    if not href.startswith('http'):
                        href = f"https://www.angelone.in{href}"
                    links.append(href)
            
            return list(set(links))  # Remove duplicates
        except Exception as e:
            print(f"Error getting support links: {str(e)}")
            return []

    def process_web_content(self, url: str) -> List[Document]:
        """Process content from a web page."""
        logger.info(f"Processing web content from: {url}")
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from web")
            
            # Log a sample of the content
            for i, doc in enumerate(docs[:2]):  # Log first 2 documents
                logger.info(f"Sample content from web document {i+1}:")
                logger.info(f"{doc.page_content[:200]}...")
            
            split_docs = self.text_splitter.split_documents(docs)
            logger.info(f"Split into {len(split_docs)} chunks")
            
            return split_docs
            
        except Exception as e:
            logger.error(f"Error processing web content: {str(e)}")
            return []

    def process_all_support_pages(self) -> List[Document]:
        """Process all support pages from the Angel One support website."""
        documents = []
        links = self.get_support_links()
        
        print(f"Found {len(links)} support pages to process")
        
        for link in links:
            try:
                docs = self.process_web_content(link)
                documents.extend(docs)
                print(f"Successfully processed: {link}")
                time.sleep(1)  # Be nice to the server
            except Exception as e:
                print(f"Error processing {link}: {str(e)}")
        
        return documents

    def initialize_vector_store(self, docs: List[Document]):
        """Initialize the vector store with documents."""
        logger.info("Initializing vector store")
        try:
            self.vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory="./data/chroma"
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def process_all_documents(self, doc_dir: str, web_url: str):
        """Process all documents and initialize the vector store."""
        logger.info("Starting document processing")
        
        # Process local documents
        docs = self.process_documents(doc_dir)
        
        # Process web content
        web_docs = self.process_web_content(web_url)
        docs.extend(web_docs)
        
        # Initialize vector store
        if docs:
            self.initialize_vector_store(docs)
        else:
            logger.warning("No documents processed successfully")

    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """Get relevant documents for a query."""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
            
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(docs)} relevant documents for query: {query}")
            
            # Log the sources of the retrieved documents
            for i, doc in enumerate(docs):
                logger.info(f"Document {i+1} source: {doc.metadata.get('source', 'Unknown')}")
                logger.info(f"Document {i+1} content preview: {doc.page_content[:200]}...")
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return [] 