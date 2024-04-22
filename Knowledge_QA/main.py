from typing import Optional, List
from langchain import PromptTemplate
from get_vector import *
from model import ChatGLM2

# 加载FAISS向量库
EMBEDDING_MODEL = r'F:\moka-ai\m3e-base'
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
FAISS.allow_dangerous_deserialization = True
db = FAISS.load_local('faiss/product', embeddings, allow_dangerous_deserialization=True)


def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content.replace('\n\n', '\n'))
    return '\n'.join(related_content)


def define_prompt():
    question = '我身高170,体重140斤,买多大尺码'
    docs = db.similarity_search(question, k=1)
    related_content = get_related_content(docs)
    PROMPT_TEMPLATE = '''
    基于以下已知信息,简介和专业的来回答用户的问题,不允许答案中添加编造的成分。
    已知内容:
    {context}
    问题:
    {question}
    '''
    prompt = PromptTemplate(
        input_variables=['context', 'question'], template=PROMPT_TEMPLATE
    )
    my_pmt = prompt.format(context=related_content, question=question)
    return my_pmt


def qa():
    llm = ChatGLM2()
    llm.load_model(r'F:\chatglm-6b')
    my_pmt = define_prompt()
    result = llm(my_pmt)
    return result


if __name__ == '__main__':
    result = qa()
    print(result)
