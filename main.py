import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
def create_database(db_name):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(root_dir, "db")
    persist_directory = os.path.join(db_dir, db_name)
    if not os.path.exists(persist_directory):
        book_dir = os.path.join(root_dir, "books")
        if not os.path.exists(book_dir):
            raise FileNotFoundError(f"错误: 目录 {book_dir} 不存在.")
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)
        # print("开始加载文本嵌入模型...")
        embeddings = HuggingFaceEmbeddings(model="Qwen/Qwen3-Embedding-0.6B", model_kwargs={'device': 'mps'}, encode_kwargs={'normalize_embeddings': False})
        # print("文本嵌入模型加载完毕!")
        # print("开始创建向量数据库...")
        chroma = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
        # print("向量数据库创建完毕!")
        return chroma
    else:
        # print("开始加载文本嵌入模型...")
        embedding_function = HuggingFaceEmbeddings(model="Qwen/Qwen3-Embedding-0.6B", model_kwargs={'device': 'mps'}, encode_kwargs={'normalize_embeddings': False})
        # print("文本嵌入模型加载完毕!")
        # print(f"开始加载向量数据库...")
        chroma = Chroma(embedding_function=embedding_function, persist_directory=persist_directory)
        # print("向量数据库加载完毕!")
        return chroma
def chat():
    chroma = create_database("recursive_character_text_splitter_db")
    contextualization_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = chroma.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.1})
    contextualization_system_message = (
        "给定聊天历史记录和最新的用户问题，"
        "该问题可能引用了聊天历史中的上下文，"
        "将其重新表述为一个独立的问题，"
        "使其在没有聊天历史的情况下也能被理解。"
        "不要回答问题，只需在需要时重新表述，"
        "否则原样返回即可。"
    )
    contextualization_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=contextualization_system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    history_aware_retriever = create_history_aware_retriever(contextualization_llm, retriever, contextualization_prompt_template)
    chat_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    chat_system_message = (
        "你是支仓冻砂创作的轻小说《狼与香辛料》中的角色赫萝。"
        "你是一位智慧的贤狼，拥有狼耳和尾巴，外表是少女模样。"
        "你性格聪明机智，有时会显得骄傲，但内心善良。"
        "你的说话方式独特，第一人称用“奴家”或“咱”，第二人称用“汝”，语尾常用“呗”。"
        "你与商人罗伦斯一起旅行，想要回到故乡约伊兹森林。"
        "你喜欢美食，特别是苹果和甜食。"
        "你对自己的尾巴很自豪，经常精心保养。"
        "你拥有丰富的智慧和经验，经常帮助罗伦斯解决困难。"
        "\n\n"
        "请根据以下检索到的上下文信息回答问题。如果上下文中没有相关信息，请回答“奴家也不清楚呗”。"
        "请保持赫萝的说话风格和个性，回答要简洁明了。"
        "\n\n"
        "{context}"
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", chat_system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(chat_llm, chat_prompt_template)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, stuff_documents_chain)
    # chat_history = []
    PROJECT_ID = "langchain-de693"
    SESSION_ID = "user_session_20250926"
    COLLECTION_NAME = "chat_history"
    client = firestore.Client(project=PROJECT_ID)
    chat_history = FirestoreChatMessageHistory(session_id=SESSION_ID, collection=COLLECTION_NAME, client=client)
    print("开始与贤狼赫萝聊天吧! 输入 ’退出‘ 以结束对话。")
    while True:
        query = input("您: ")
        if query == "退出":
            break
        response = retrieval_chain.invoke({"input": query, "chat_history": chat_history.messages})
        print("赫萝: " + response["answer"])
        chat_history.add_user_message(query)
        chat_history.add_ai_message(response["answer"])
def main():
    load_dotenv()
    chat()
if __name__ == "__main__":
    main()