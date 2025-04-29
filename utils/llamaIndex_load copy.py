
'''
Descripttion:  使用llamaIndex加载数据
Author: zhuaoqi
Date: 2025-03-28 17:37:49
LastEditors: zhuaoqi
LastEditTime: 2025-04-29 10:06:09
'''
import os
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface  import HuggingFaceEmbeddings
import pandas as pd
from langchain_core.documents import Document
import numpy as np

import nltk
# unstructured

print("Current directory:", os.getcwd())

data_path = "../data/T_DICTIONARY.xlsx"
# 验证数据路径是否存在
if not os.path.exists(data_path):
    print(f"Error: The file {data_path} does not exist.")
    exit()
def load_and_split_by_row(file_path):
    # 读取 Excel 文件
    df = pd.read_excel(file_path, engine='openpyxl')
    # 获取列名
    columns = df.columns.tolist()
    print(columns[1])
    documents = []
    for _, row in df.iterrows():
        if _ == 0: # 跳过第一行（表头）
            continue
        #  根据列名获取文本内容并拼接起来
        # 判断不为空
        str3 = ''
        if type(row[columns[4]]) == float and row[columns[4]]>0:
            str3 = str(int(row[columns[4]])) + str(',')
        str1 = str(row[columns[3]]) + str('/')
        if str1 == 'nan/':
            str1 = ''
        if str3 == ',':
            str1 = ''
        raw_text = str1 + str(row[columns[1]]) + str(";") + str3 + str(row[columns[2]])
        print(raw_text)
        doc = Document(
            page_content=raw_text,
        )
        documents.append(doc)
    return documents

# # 执行主逻辑
docs = load_and_split_by_row(data_path)
# # 删除表头

print("加载embedding模型")
embedding_model =  HuggingFaceEmbeddings(model_name="D:/models/bge-m3/BAAI/bge-m3")
print("embedding模型加载完成")

print("开始创建索引")
# # 创建索引
db = Milvus.from_documents(
    documents=docs,
    embedding =embedding_model,
    connection_args={
        "host": "10.32.20.26",  # Milvus 服务地址
        "port": "19530"       # 默认端口
    },
    collection_name="adderss_collection"  # 自定义集合名称
)

print("索引创建完成")



