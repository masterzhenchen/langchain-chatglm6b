from typing import Optional, List
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings import QianfanEmbeddingsEndpoint

def main():
    EMBEDDING_MODEL=QianfanEmbeddingsEndpoint()
    # 定义向量模型路径
    # EMBEDDING_MODEL = r'F:\moka-ai\m3e-base'
    # 加载文档
    loader = UnstructuredFileLoader('衣服属性.txt')
    data = loader.load()
    print(f'documents:{len(data)}')
    # 第二部:切分文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    # 切割加载的document
    split_docs = text_splitter.split_documents(data)
    # 第三步:初始化huggingface的embedding对象
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # 第四步:将document通过embedding对象计算得到的向量信息并永久存入FAISS向量数据库,用于后续匹配查询
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local('faiss/product')
    return split_docs



