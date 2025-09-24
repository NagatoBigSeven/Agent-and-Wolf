import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
load_dotenv()
root_dir = os.path.dirname(os.path.abspath(__file__))
book_dir = os.path.join(root_dir, "books")
db_dir = os.path.join(root_dir, "db")
if not os.path.exists(book_dir):
    raise FileNotFoundError(f"Error: Directory {book_dir} does not exist.")
documents = []
for book_name in os.listdir(book_dir):
    if not book_name.endswith(".txt"):
        continue
    book_path = os.path.join(book_dir, book_name)
    text_loader = TextLoader(book_path)
    book_documents = text_loader.load()
    for book_document in book_documents:
        book_document.metadata["source"] = book_name
    documents += book_documents
def create_database(documents, embedding, db_name):
    persist_directory = os.path.join(db_dir, db_name)
    if not os.path.exists(persist_directory):
        result = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
        return result
    else:
        print(f"Database {db_name} already exists.")
def query_database(query, embedding_function, db_name):
    persist_directory = os.path.join(db_dir, db_name)
    if os.path.exists(persist_directory):
        database = Chroma(embedding_function=embedding_function, persist_directory=persist_directory)
        retriever = database.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.1})
        relevant_docs = retriever.invoke(query)
        for i, relevant_doc in enumerate(relevant_docs):
            print(f"doc {i + 1}:\n{relevant_doc.page_content}")
            if relevant_doc.metadata:
                print(f'source: {relevant_doc.metadata.get("source", "Don't know")}')
        return relevant_docs
    else:
        raise FileNotFoundError(f"Error: Database {db_name} does not exist.")
def main():
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': False}
    print("开始加载文本嵌入模型...")
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    print("文本嵌入模型加载完毕!")
    '''
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    create_database(docs, embeddings, "character_text_splitter_db")
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    create_database(docs, embeddings, "token_text_splitter_db")
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    create_database(docs, embeddings, "sentence_transformers_token_text_splitter_db")
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print("开始创建向量数据库...")
    create_database(docs, embeddings, "recursive_character_text_splitter_db")
    print("向量数据库创建完毕!")
    query = "罗伦斯和赫萝结婚了吗？"
    '''
    query_database(query, embeddings, "character_text_splitter_db")
    query_database(query, embeddings, "token_text_splitter_db")
    query_database(query, embeddings, "sentence_transformers_token_text_splitter_db")
    '''
    print("开始查询向量数据库...")
    query_database(query, embeddings, "recursive_character_text_splitter_db")
    print("向量数据库查询完毕!")
if __name__ == "__main__":
    main()