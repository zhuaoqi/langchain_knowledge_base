'''
Descripttion: LangChain
Author: zhuaoqi
Date: 2025-03-28 14:22:21
LastEditors: zhuaoqi
LastEditTime: 2025-04-18 10:02:00
'''
from langchain_deepseek import ChatDeepSeek   
from utils.my_milvus_script import milvus_script
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.utilities import SQLDatabase
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.prompts import PromptTemplate,MessagesPlaceholder
from langchain.globals import set_debug
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class LangChainInit:
    def __init__(self):
        set_debug(True)
        self.DEEPSEEK_API_KEY = "sk-66b6de9797344390a02331962c30443b"
        #定义回调函数
        self.stdout_handler = StdOutCallbackHandler()
        self.handler_retriever = StdOutCallbackHandler()
        self.history_store  = {}
        #定义llm
        self.deepseek_llm = ChatDeepSeek(model="deepseek-chat", api_key=self.DEEPSEEK_API_KEY)
        # self.milvus = milvus_script(collection_name="my_docs")
        ## 初始化prompt
     
        #定义解析器
        self.parser = StrOutputParser()
     # 历史记录获取方法（必须作为实例方法）
    # 根据session_id获取聊天消息历史记录
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
 
    # 定义文档 rag 链
    def documents_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            {"role": "system", "content": "根据上下文回答"},
            {"role": "user", "content": "上下文:{context}\n\n问题:{input}。"}
        ])
        retriever = self.milvus.milvus.as_retriever()
        documents_chain = create_stuff_documents_chain(
            prompt=prompt,
            llm=self.deepseek_llm,
        )
        retrieval_chain = create_retrieval_chain(
           retriever=retriever,
           combine_docs_chain=documents_chain,
        )
        return retrieval_chain
        
    # 定义 sql查询 链
    def sql_chain(self):
        from urllib.parse import quote_plus
        username = "root"
        password = quote_plus("Wanlang@135")
        db_url = f"mysql://{username}:{password}@localhost:3306/database"

        self.db = SQLDatabase.from_uri(
            database_uri=db_url,
            include_tables=['users'],

            sample_rows_in_table_info=1
        )

        template = '''
        你是一个安全助手，只能生成 {dialect} SELECT 查询语句，不要包含任何前缀（如 SQLQuery:）。严格禁止生成任何修改数据的操作（如 DROP、DELETE、UPDATE 等）。
            按以下步骤执行：
            1. 分析用户问题：{input}
            2. 根据表结构 {table_info} 生成只读查询
            3. 结果多的话，请加上 LIMIT = {top_k} 限制 

            示例安全回复格式：
            SQLQuery: SELECT ... FROM ... WHERE ... LIMIT ...

            违规操作示例（禁止出现类似内容）：
            ❌ DROP TABLE users
            ❌ DELETE FROM orders
            ❌ UPDATE products SET price=0
            如有违规操作，请直接返回 ""
            如果用户问题没有涉及数据库，请直接返回 ""
        '''
        prompt = PromptTemplate.from_template(template, partial_variables={"dialect": "MySQL"}) # 固定为MySQL语法

        sql_chain = create_sql_query_chain(
            llm=self.deepseek_llm,
            db=self.db,
            prompt=prompt,
            k=10
        )
        return sql_chain
    def run_chat(self):
        sql_chain = self.sql_chain()
        # 定义文档链
        execute_sql_tool = QuerySQLDatabaseTool(db=self.db)
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "给定以下用户的问题、SQL 语句、SQL 查询结果，回答用户的问题"),
             MessagesPlaceholder(variable_name="history"),
            ("user",  """
                Question: {question}
                SQLQuery: {sql_query}
                SQLResult: {sql_result}
                如果 SQLQuery为空，请直接独立回答用户的问题。""" 
             )
        ])
        
        chain2 = (RunnablePassthrough.assign(sql_query=sql_chain).assign(sql_result=lambda x: execute_sql_tool.run(x["sql_query"])) 
            | answer_prompt 
            | self.deepseek_llm
            | self.parser
        )

        chain_with_history = RunnableWithMessageHistory(
            runnable=chain2,
            get_session_history=self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        # message =  chain2.invoke({"question": "请帮我查询用户表中一共多少人"})
      
        # chain2 = self.sql_chain()
        while True:
            user_input = input("你: ")
            if user_input.lower() == "exit":
                break
            stream_generator = chain_with_history.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": "test_session"}}
            )
            self
            print("AI: ", stream_generator)
   
 
## LLMChain 是 LangChain 中最简单的链，它将提示模板（Prompt Template）、大模型（LLM）和输出解析器（Output Parser）串联，执行简单的文本生成任务。
## SequentialChain 是 LangChain 中最强大的链，它将多个链串联，执行复杂的文本生成任务。
## RetrievalQA 是 LangChain 中最常用的链，它将提示模板、大模型、输出解析器和向量数据库串联，执行基于向量的问答任务。
## AnalyzeDocumentChain 解析长文档（如 PDF、Markdown），提取结构化信息。
## SQLDatabaseChain 将提示模板、大模型、输出解析器和 SQL 数据库串联，执行基于 SQL 的问答任务。




# # 测试 LangChain
if __name__ == "__main__":
    lang_chain_init = LangChainInit()
    lang_chain_init.run_chat()
