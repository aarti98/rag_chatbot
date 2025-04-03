from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .document_processor import DocumentProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={
                "temperature": 0.05,
                "max_length": 2048,
                "top_p": 0.9,
                "do_sample": False 
            }
        )
        self.setup_chat_prompt()

    def setup_chat_prompt(self):
        """Set up the chat prompt template."""
        template = """You are a support assistant for Angel One. Your task is to answer questions based on the information provided in the context below.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the information provided in the context above
2. If the context contains relevant information, provide a detailed answer with specific details
3. If the context doesn't contain enough information to answer the question, say "I don't know"
4. Do not make up information or use external knowledge
5. If you find multiple relevant pieces of information, combine them into a comprehensive answer
6. Keep your answer concise and relevant
7. If the question is about what is covered under a plan, list the specific coverage details mentioned in the context

Answer:"""

        self.prompt = ChatPromptTemplate.from_template(template)

    def format_docs(self, docs):
        """Format documents for the prompt."""
        formatted_docs = []
        seen_content = set()  # Track unique content to avoid duplicates
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown source')
            content = doc.page_content.strip()
            
            # Skip if we've seen this content before
            content_hash = hash(content)
            if content_hash in seen_content:
                continue
                
            seen_content.add(content_hash)
            
            # Add page number if available
            page = doc.metadata.get('page', '')
            page_info = f" (Page {page})" if page else ""
            formatted_docs.append(f"Source {i} ({source}{page_info}):\n{content}\n")
            
        return "\n".join(formatted_docs)

    def get_response(self, query: str) -> str:
        """Get response for a query."""
        try:
            # Get relevant documents
            docs = self.document_processor.get_relevant_documents(query)
            
            if not docs:
                return "I don't know"
            
            # Format documents for the prompt
            formatted_docs = self.format_docs(docs)
            
            # Log the context for debugging
            logger.info(f"Context being used:\n{formatted_docs[:500]}...")
            
            # Create the chain
            chain = self.prompt | self.llm | StrOutputParser()
            
            # Get response
            response = chain.invoke({"context": formatted_docs, "question": query})
            
            # Log the response for debugging
            logger.info(f"Query: {query}")
            logger.info(f"Response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return "I don't know"

    def initialize(self, doc_dir: str, web_url: str):
        """Initialize the chatbot with documents."""
        try:
            self.document_processor.process_all_documents(doc_dir, web_url)
            return True
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            return False 