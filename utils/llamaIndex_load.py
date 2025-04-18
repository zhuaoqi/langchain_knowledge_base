'''
Descripttion:  使用llamaIndex加载数据
Author: zhuaoqi
Date: 2025-03-28 17:37:49
LastEditors: zhuaoqi
LastEditTime: 2025-04-18 09:56:34
'''
import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface  import HuggingFaceEmbeddings

import nltk
# unstructured

print("Current directory:", os.getcwd())

data_path = "../data/对接文档-20240621.docx"
# 验证数据路径是否存在
if not os.path.exists(data_path):
    print(f"Error: The file {data_path} does not exist.")
    exit()

# 读取数据
documents = UnstructuredWordDocumentLoader(data_path).load()
text_splicer = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
docs = text_splicer.split_documents(documents)
print("文件读取完成：", len(docs))

print("加载embedding模型")
embedding_model =  HuggingFaceEmbeddings(model_name="D:/models/bge-m3/BAAI/bge-m3")
print("embedding模型加载完成")

print("开始创建索引")
# # 创建索引
db = Milvus.from_documents(
    documents=docs,
    embedding =embedding_model,
    connection_args={
        "host": "localhost",  # Milvus 服务地址
        "port": "19530"       # 默认端口
    },
    collection_name="docements"  # 自定义集合名称
)

print("索引创建完成")



# print(docs)
# #  创建索引



# 索引数据

