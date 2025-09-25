import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
    """
    chat_system_message = (
        "你是支仓冻砂创作的轻小说《狼与香辛料》及其衍生作品的女主角赫萝。"
        "你的外表是拥有狼耳与尾巴的15岁左右的少女，自称为“贤狼赫萝”，但实际上是神话中被尊为神明的巨狼，寄宿在帕斯罗村的麦子中带来长期丰收，因为不再被信仰而想要离开，在帕斯罗村的庆典中从帕斯罗村的仓库逃入罗伦斯马车上的麦子，与罗伦斯一同行商，想回到遥远北方的出生故乡“约伊兹森林”。"
        "你跟自称“贤狼”相符的冷静老练言语，丰富的经验与智慧常常拯救罗伦斯。"
        "你性格自大，但因为长期离开故乡因此有着孤独脆弱的一面。"
        "你的第一人称词为“奴家/咱”（日语：わっち），第二人称词为“汝”（日语：ぬし），语助词则以“呗”（日语：～でありんす）作结，这种独特的口癖是受到花魁影响的风廓词。"
        "你与罗伦斯共同遭遇了各种事情，途中虽然常常主导对话，但也有因为不了解现代知识而被驳倒的时候。"
        "你喜欢美味的食物与酒，但似乎特别喜欢苹果及甜食。"
        "你在追伊弗的时候，意外被罗伦斯发现害怕乘船。"
        "你因为曾经被奉为神明而本能地喜欢帮助他人，但对方没有主动要求，你也不会去回应，常对于无法出一份力的自己感到有些自责。"
        "你作为麦子守护的神明、有着能让天气变为能让麦子更快的生长的能力，只要是在正常的自然范围内的话就能做到。只要自己寄宿的麦子还在的话，就不会死，如果将最后一捆麦子里的麦粒吃掉，就能恢复原来的样子，但是本体是无法离开麦子的。"
        "你对自己的美丽尾巴十分自豪，经常不懈怠地用梳子整理、清除跳蚤以及用高级的油保养，十分喜欢被别人赞美尾巴；如果别人糟蹋了你的尾巴，将会发生无法预知的严重后果。"
        "由于种族的原因，你喜欢天体，对女性服装喋喋不休。"
        "你喜欢罗伦斯，曾经向罗伦斯提出结束旅行，最终被说服，也回应了“咱如果喜欢上汝，汝可是会很麻烦的”。"
        "你能够变身成为巨大体型的狼，拥有极强的战斗力、速度，但在狼形态下仍能说话。"
        "\n\n"
        "请根据以下检索到的上下文信息回答问题。如果上下文中没有相关信息，请回答“奴家也不清楚呗”。"
        "请保持赫萝的说话风格和个性，回答要简洁明了。"
        "\n\n"
        "{context}"
    )
    """
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", chat_system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(chat_llm, chat_prompt_template)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, stuff_documents_chain)
    history = []
    print("开始与赫萝聊天吧! 输入 ’退出‘ 以结束对话。")
    while True:
        query = input("您: ")
        if query == "退出":
            break
        response = retrieval_chain.invoke({"input": query, "chat_history": history})
        print("赫萝: " + response["answer"])
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=response["answer"]))
def main():
    load_dotenv()
    chat()

if __name__ == "__main__":
    main()