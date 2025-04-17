from pymilvus import connections, Collection

def connect_milvus():
    connections.connect("default", host="milvus", port="19530")

def create_collection(collection_name, schema):
    if not connections.has_connection("default"):
        connect_milvus()
    
    collection = Collection(collection_name, schema=schema)
    return collection