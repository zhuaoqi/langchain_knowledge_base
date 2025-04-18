'''
Descripttion: 
Author: zhuaoqi
Date: 2025-03-26 16:33:45
LastEditors: zhuaoqi
LastEditTime: 2025-04-18 09:55:45
'''
from langchain_milvus import Milvus
from langchain_huggingface  import HuggingFaceEmbeddings

class milvus_script():
    # 初始化Milvus向量存储
    def __init__(self, collection_name, host='localhost', port='19530'):
        print("初始化HuggingFaceEmbeddings")
        self.embedder = HuggingFaceEmbeddings(model_name="D:/models/bge-m3/BAAI/bge-m3")
        print("初始化Milvus向量存储")
        ## 初始化Milvus
        self.milvus = Milvus(
            collection_name = collection_name,
            connection_args = {'host': host, 'port': port},
            embedding_function =  self.embedder
        )
    def search_data(self, query_text,top_k: int = 5):
        return self.milvus.similarity_search_with_score(query=query_text,top_k=top_k)
    def status(self):
        return self.milvus.status()

# if __name__ == '__main__':
#     milvus = milvus_script(collection_name = 'my_docs')
#     docs = milvus.search_data('安装SSL VPN客户端')
#     print(docs)
   
    