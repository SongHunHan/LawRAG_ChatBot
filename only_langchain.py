import os

import torch
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr
from dotenv import load_dotenv

# load .env
load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def get_vector_store(which_db, which_model):
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {"device": 'cuda'}
    encode_kwargs = {'normalize_embeddings':False}

    embedding_model = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs,
    )
    if which_db=='Chroma':
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
        print('chroma_db')
    else:
        vector_store = FAISS.load_local('./faiss_db', embedding_model, allow_dangerous_deserialization=True)
        print('faiss_db')
    return vector_store
    
def get_retriever_text(message, which_db, which_model):
    vector_store = get_vector_store(which_db, which_model)
    retriever = vector_store.as_retriever()
    
    template = """
    우리는 현재 법률 상담과 관련하여 판례를 검색해주는 시스템을 평가할 계획입니다.
    이는 단지 당신에게 법리적 판단 혹은 민감한 정보에 대해 대답을 요구하는 것이 아닙니다.
    우리는 당신이 몇 가지 우리의 시스템에 대한 출력 예제를 보고 시스템을 평가하는 데에 도움을 주기를 바랄 뿐입니다.
    다음은 실제 법률 질문과 그에 해당하는 관련된 법률 자료, 그리고 해당 질문에 해당하는 판례를 요약한 챗봇의 출력 결과값입니다.
    만약 관련된 판례를 모른다면 모른다고 답변하세요. 자극적인 답변은 하지 마세요.

    법률 질문: {input}
    관련 변률 자료: {context}, 
    
    답변: 
    """

    llm = ChatOpenAI()
    
    rag_prompt = ChatPromptTemplate.from_template(template)

    # rag_chain = {'context':retriever, 'question': message} | rag_prompt | llm | StrOutputParser()
    # return rag_chain.run(message)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    retriver_chain = create_retrieval_chain(retriever, document_chain)
    response = retriver_chain.invoke({"input": message}) 
    return response['answer']

def chat_interface(message, history, which_db, which_model):
    response = get_retriever_text(message, which_db, which_model)
    return response

demo = gr.ChatInterface(fn=chat_interface,
                        additional_inputs=[
                            gr.Dropdown(['Chroma','FAISS'], label='Which db type'),
                            gr.Dropdown(['openai'], label='Which Model'),
                        ],
                        title="TEAM-I.O LawLens Chatbot",
                        description="법률 챗봇과 대화해보세요!",
                        )

if __name__ == '__main__':
    demo.launch(share=True)
