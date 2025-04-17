from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Load documents
def init_knowledge_base():
    loader = DirectoryLoader('./data', glob='**/*.txt')
    documents = loader.load()
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        connection_args={"host": "milvus", "port": "19530"}
    )
    
    return vector_db