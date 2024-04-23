from typing import Optional, List
import os
from langchain import PromptTemplate
from langchain_community.llms import QianfanLLMEndpoint

# 因为是langchain框架,所以这里处理文本数据都是基于langchain框架来进行处理的
os.environ['QIANFAN_AK'] = '2jCwbiCWReSVlPbt53f9pYhP'
os.environ['QIANFAN_SK'] = 'dwJ8BULLCsBMXsmtjyxNI8hkw4y0HPXq'
# 先把要查询的句子进行词向量编码
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS


def getdb():
    # 加载文档
    data = UnstructuredFileLoader('衣服属性.txt').load()
    # 切分文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.split_documents(data)
    # 初始化huggingface的embedding对象
    embeddings = QianfanEmbeddingsEndpoint()
    db = FAISS.from_documents(split_docs, embeddings)
    return db


# 文档加载进FAISS数据库已经加载完毕,
# 定义提取相关文档内容的函数
def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content.replace('\n\n', '\n'))
    return '\n'.join(related_content)  # 返回拼接后的相关内容字符串


# 定义提示模板的函数
def define_prompt(question: str, db: FAISS):
    docs = db.similarity_search(question, k=1)
    related_content = get_related_content(related_docs=docs)
    PROMPT_TEMPLATE = '''
    基于以下已知信息,简介和专业的来回答用户的问题,不允许答案中添加编造的成分。
    已知内容:
    {context}  # 这里使用了字符串插值，将相关内容插入到模板中
    问题:
    {question}  # 这里同样使用了字符串插值，将问题插入到模板中
    '''
    prompt = PromptTemplate(
        input_variables=['context', 'question'], template=PROMPT_TEMPLATE
    )
    # 格式化模板,替换变量为实际内容
    my_mpt = prompt.format(context=related_content, question=question)
    return my_mpt


# 设置QA即问答函数
def question_answer(question: str):
    llm = QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-7B')
    my_pmt = define_prompt(question, getdb())
    return llm(my_pmt)


if __name__ == '__main__':
    question = '我身高170,体重140斤,买多大尺码'
    print(question_answer(question))
