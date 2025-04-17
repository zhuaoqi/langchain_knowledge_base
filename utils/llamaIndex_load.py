from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

def load_documents():
    documents = SimpleDirectoryReader('./data').load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    return index