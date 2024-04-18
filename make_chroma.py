import os

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

def return_json_path():
    val_path = '/workspace/ssd/AI_hub/law_data/data/2.Validation/labeled_data/VL_1.판결문/2.Validation/라벨링데이터/VL_1.판결문'
    train_path = "/workspace/ssd/AI_hub/law_data/data/1.Training/labeled_data/TL_1.판결문/TL_1.판결문/1.Training/라벨링데이터/TL_1.판결문"

    json_file_path = []
    for root,dirs,files in os.walk(val_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                json_file_path.append(file_path)
    for root,dirs,files in os.walk(train_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                json_file_path.append(file_path)
    documents = []

    for file_path in json_file_path:
        loader = JSONLoader(
        file_path=file_path,
        jq_schema='.',
        text_content=False,
        )

        loaded_docs = loader.load()
        documents.extend(loaded_docs)

    encoded_docs = []

    for idx, doc in enumerate(documents):
        broken_korean = doc.page_content
        fixed_korean = broken_korean.encode('latin1').decode('unicode-escape')
        
        # refined_doc = Document(page_content=fixed_korean)
        refined_doc = Document(page_content=fixed_korean, metadata={'source': doc.metadata['source'].split('/')[-1]}) 
        encoded_docs.append(refined_doc)

    return encoded_docs

def make_db(encoded_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64
    )
    splits = text_splitter.split_documents(documents=encoded_docs)

    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {"device": 'cuda'}
    encode_kwargs = {'normalize_embeddings':False}   
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs,
    )

    vector_storage = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory="./chroma_db")

    ## for test 
    print(vector_storage.similarity_search('가장 비싼 내용'))

make_db(return_json_path())