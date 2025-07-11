# pip install agno ollama fastembed
# pip install langchain-community langchain
# pip install langchain-qdrant

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant.qdrant import QdrantVectorStore
from qdrant_client import qdrant_client
from qdrant_client.http.models import Distance,VectorParams 
from qdrant_client.http.exceptions import UnexpectedResponse

# agentic workflow (make sure ollama is installed locally to run this)
from agno.agent import Agent
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.models.ollama import Ollama

# Building the Retrievel pipeline

urls = ["https://www.uber.com/en-IN/blog/reinforcement-learning-for-modeling-marketplace-balance/?uclick_id=ae30a79d-1208-4caa-a3f1-7231a4562fba"]
loader = WebBaseLoader(urls)
data = loader.load()

# once you have the URL pass it inside the webbaseloader
# this gives you raw document or the document objects

# convert that into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 50)
chunks  = text_splitter.split_documents(data)

# once the chunks are ready we use the embedding model from FastEmbed

embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")
                                
# once above things are defined, we need to setup of qdrant database (in -memoery) 

# we need to set the path where we save the embedding model
client = qdrant_client(path = "/tmp/app")
# define collection : collection name needs to be unique for each data change
# or updates
collection_name = "agent-rag"

# When we create a RAG pipeline , we need to index the documents (this is a 1 time process)
# the next time, when you are asking the same quert from the document
# you want to avoid re-indexing it again 
# this is why to avoid indexing again , we define unique collection again

try:
    collection_info = client.get_collection(collection_name = collection_name)
except (UnexpectedResponse,ValueError):
    client.create_collection(
        collection_name = collection_name,
        vector_config = VectorParams(size = 1024, distance= Distance.COSINE),
        )

vector_store =QdrantVectorStore(
    client = client, 
    collection_name= collection_name,
    embedding= embeddings,
)

# once the vector store is ready , add all chunks to the vector store
vector_store.add_documents(documents=chunks)


# We need to define retriever that is our knowledge base
retriever = vector_store.as_retriever()
# once we have retriver in place, we define knowledge base
knowledge_base = LangChainKnowledgeBase(retriever= retriever)

# the retriever pipeline is ready

# now we will build the agentic workflow
agent = Agent(
    model = Ollama(id = 'llama3.2:1b'),
    knowledge= knowledge_base,
    description= "Answer the user queries from the knowledge base. \
        The answers should be pertinent with the question asked.\
            Avoid any unnecessary information.",
    markdown = True,
    search_knowledge= True, # this helps interact with the knowledge base
    )

# now we have defined the agent
# we need to execute the agentic RAG by defining a user query

user_query = "Can you throw some llight on the Reward modelling for unutilized states ? "

agent.print_response(user_query,stream=True)

# alternate method to extract output
# response = agent.run(user_query).content
# print(response)