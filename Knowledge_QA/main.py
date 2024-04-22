# 导入必要的模块和函数
from typing import Optional, List  # 导入类型提示所需的模块
from langchain import PromptTemplate  # 导入提示模板
from get_vector import *  # 导入从向量获取相关内容的函数
from model import ChatGLM2  # 导入聊天模型

# 加载FAISS向量库
EMBEDDING_MODEL = r'F:\moka-ai\m3e-base'  # 设定嵌入模型的路径
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # 实例化 HuggingFaceEmbeddings 类，并加载指定的模型
FAISS.allow_dangerous_deserialization = True  # 允许 FAISS 进行危险的反序列化操作
db = FAISS.load_local('faiss/product', embeddings, allow_dangerous_deserialization=True)  # 加载本地 FAISS 数据库

# 提取相关文档内容的函数
def get_related_content(related_docs):
    related_content = []  # 初始化一个空列表，用于存储相关内容
    for doc in related_docs:  # 遍历相关文档列表
        related_content.append(doc.page_content.replace('\n\n', '\n'))  # 将文档内容添加到列表中，替换掉多余的空行
    return '\n'.join(related_content)  # 返回拼接后的相关内容字符串

# 定义提示模板的函数
def define_prompt():
    question = '我身高170,体重140斤,买多大尺码'  # 设置要询问的问题
    docs = db.similarity_search(question, k=1)  # 使用问题进行相似性搜索，返回最相似的文档
    related_content = get_related_content(docs)  # 提取相关文档的内容
    PROMPT_TEMPLATE = '''
    基于以下已知信息,简介和专业的来回答用户的问题,不允许答案中添加编造的成分。
    已知内容:
    {context}  # 这里使用了字符串插值，将相关内容插入到模板中
    问题:
    {question}  # 这里同样使用了字符串插值，将问题插入到模板中
    '''
    prompt = PromptTemplate(  # 创建一个 PromptTemplate 实例
        input_variables=['context', 'question'], template=PROMPT_TEMPLATE  # 指定输入变量和模板内容
    )
    my_pmt = prompt.format(context=related_content, question=question)  # 格式化模板，替换变量为实际内容
    return my_pmt  # 返回格式化后的提示内容

# QA（问答）函数
def qa():
    llm = ChatGLM2()  # 创建 ChatGLM2 实例
    llm.load_model(r'F:\chatglm-6b')  # 加载 ChatGLM2 模型
    my_pmt = define_prompt()  # 获取格式化后的提示
    result = llm(my_pmt)  # 使用模型进行问答
    return result  # 返回问答结果

# 如果作为主程序运行
if __name__ == '__main__':
    result = qa()  # 执行问答函数
    print(result)  # 打印问答结果