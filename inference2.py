import warnings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Agentic workflow libraries
from agno.agent import Agent
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.models.ollama import Ollama

# Suppress the specific UserWarning from httpx
warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set")

# --- 1. Load and Chunk Documents ---
print("Step 1: Loading and splitting documents...")
urls = ["https://www.uber.com/en-IN/blog/reinforcement-learning-for-modeling-marketplace-balance/"]
loader = WebBaseLoader(urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
print(f"Successfully split data into {len(chunks)} chunks.")

# --- 2. Setup Embeddings Model ---
# We use OllamaEmbeddings since Ollama is already running for the agent.
# This simplifies dependencies and ensures consistency.
print("Step 2: Initializing embeddings model...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# --- 3. Setup Qdrant Vector Store ---
# This code block sets up a local, in-memory Qdrant vector store.
# The 'from_documents' method is a convenient way to create and populate the store in one step.
collection_name = "agent-rag-uber-rl"
qdrant_path = "/tmp/qdrant_db"

print(f"Step 3: Creating and populating Qdrant vector store at '{qdrant_path}'...")

# The vector size for Llama 3 is 4096. This MUST match the embedding model.
vector_store = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    path=qdrant_path,
    collection_name=collection_name,
    force_recreate=True,  # Set to False to reuse an existing collection
    vector_params=VectorParams(size=1024, distance=Distance.COSINE),
)
print("Vector store created and populated successfully.")


# --- 4. Create Retriever and Knowledge Base ---
print("Step 4: Setting up retriever and knowledge base...")
# The retriever is responsible for fetching relevant documents from the vector store.
retriever = vector_store.as_retriever()

# The LangChainKnowledgeBase wraps the retriever for use with the Agno agent.
knowledge_base = LangChainKnowledgeBase(retriever=retriever)
print("Knowledge base is ready.")

# --- 5. Build and Run the Agent ---
print("Step 5: Building the agent...")
# The agent uses the Ollama model for reasoning and the knowledge base for context.
# Note: The parameter is 'model', not 'id'.
agent = Agent(
    model=Ollama(id='qwen2.5:0.5b'),
    knowledge=knowledge_base,
    description="You are an expert assistant. Answer user queries based *only* on the provided context from the knowledge base. Be concise and accurate.",
    markdown=True,
    search_knowledge=True,  # This tells the agent to use the knowledge base.
)
print("Agent built successfully. Ready to answer questions.")

# --- 6. Ask a Question ---
user_query = "Can you throw some light on the Reward modelling for unutilized states?"

print("\n--- User Query ---")
print(user_query)
print("\n--- Agent Response ---")

# The agent will now retrieve relevant chunks, inject them as context,
# and generate an answer using Llama 3.
agent.print_response(user_query, stream=True)

# You can also get the response as a string
# response = agent.run(user_query).content
# print(response)