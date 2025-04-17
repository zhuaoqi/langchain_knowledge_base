import pymilvus

# Connect to Milvus
connections.connect("default", host="milvus", port="19530")

# Test connection
try:
    print(pymilvus.utility.get_server_version())
except Exception as e:
    print(f"Error connecting to Milvus: {e}")