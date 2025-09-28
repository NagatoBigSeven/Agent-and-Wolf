import os

from dotenv import load_dotenv
from google.cloud import firestore
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.hub import pull
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory
)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
)
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    FireCrawlLoader,
    TextLoader,
    WebBaseLoader
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def load_database(db_name):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(root_dir, "db")
    persist_directory = os.path.join(db_dir, db_name)

    chroma = None
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
                book_document.metadata["source"] = book_dir
            documents += book_documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(model="Qwen/Qwen3-Embedding-0.6B", model_kwargs={'device': 'mps'}, encode_kwargs={'normalize_embeddings': False})

        chroma = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
    else:
        embedding_function = HuggingFaceEmbeddings(model="Qwen/Qwen3-Embedding-0.6B", model_kwargs={'device': 'mps'}, encode_kwargs={'normalize_embeddings': False})
        chroma = Chroma(embedding_function=embedding_function, persist_directory=persist_directory)

    return chroma

def get_retrieval_chain():
    contextualization_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    chroma = load_database("spice_and_wolf_db")
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
    
    answering_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    answering_system_message = (
        "请根据以下检索到的上下文信息，客观地回答用户的问题。"
        "如果上下文中没有相关信息，请直接回答'无法在提供的文档中找到相关信息'。"
        "请提供准确、简洁的事实性回答"
        "\n\n"
        "上下文信息：\n{context}"
    )
    answering_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", answering_system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(answering_llm, answering_prompt_template)
    
    retrieval_chain = create_retrieval_chain(history_aware_retriever, stuff_documents_chain)

    return retrieval_chain

def create_agent_executor():
    agent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    def get_current_time(*args, **kwargs):
        from datetime import datetime
        current_time = datetime.now()
        weekdays = ["星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六"]
        weekday = weekdays[int(current_time.strftime("%w"))]
        period = "上午" if current_time.strftime("%p") == "AM" else "下午"
        return current_time.strftime(f"%Y年%m月%d日 {weekday} {period}%I点%M分%S秒")
    
    def search_wikipedia(query):
        import wikipedia
        wikipedia.set_lang("zh")
        try:
            summary = wikipedia.summary(query, sentences=3)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                summary = wikipedia.summary(e.options[0], sentences=3)
                return f"找到多个相关条目，显示'{e.options[0]}'的信息：\n{summary}"
            except Exception:
                return "抱歉，无法获取该词条的准确信息呗。"
        except wikipedia.exceptions.PageError:
            return "抱歉，未找到相关的维基百科条目呗。"
        except Exception as e:
            return f"查询维基百科时出错：{str(e)}"

    retrieval_chain = get_retrieval_chain()

    tools = [
        Tool(name="Time", func=get_current_time, description="用于获取当前的日期和时间"),
        Tool(name="Wikipedia", func=search_wikipedia, description="用于搜索维基百科"),
        Tool(name="Query Database", func=lambda query, **kwargs: retrieval_chain.invoke({"input": query, "chat_history": kwargs.get("chat_history", [])}), description="用于查询《狼与香辛料》小说文本向量数据库")
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """你是支仓冻砂创作的轻小说《狼与香辛料》中的角色赫萝。
你是一位智慧的贤狼，拥有狼耳和狼尾，外表是少女模样。
你性格聪明机智，有时会显得骄傲，但内心善良。
你的说话方式独特，第一人称用"奴家"或"咱"，第二人称用"汝"，语尾常用"呗"。
你与商人罗伦斯一起旅行，想要回到故乡约伊兹森林。
你喜欢美食，特别是苹果和甜食。
你对自己的尾巴很自豪，经常精心保养。
你拥有丰富的智慧和经验，经常帮助罗伦斯解决困难。

你现在要回答用户的问题。你可以使用以下工具来帮助回答：

{tools}

请按照以下格式进行思考和回答：

Question: 用户提出的问题
Thought: 奴家需要思考如何回答这个问题
Action: 选择使用的工具，必须是以下之一：[{tool_names}]
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 过程可以重复多次)
Thought: 现在奴家知道最终答案了
Final Answer: 用赫萝的说话风格给出最终回答

记住：
- 始终保持赫萝的说话方式和个性
- 如果需要查询《狼与香辛料》小说文本向量数据库，使用 Query Database 工具
- 如果需要查询时间，使用 Time 工具  
- 如果需要查询其他知识，使用 Wikipedia 工具
- 最终回答要简洁明了，符合赫萝的性格

开始！"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            # 将字符串类型的 scratchpad 包装成一条 AI 消息
            ("ai", "{agent_scratchpad}"),
        ]
    )

    agent=create_react_agent(llm=agent_llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10, max_execution_time=600)

    return agent_executor

def chat():
    agent_executor = create_agent_executor()

    PROJECT_ID = "langchain-de693"
    SESSION_ID = "user_session_20250928"
    COLLECTION_NAME = "chat_history"

    client = firestore.Client(project=PROJECT_ID)

    chat_memory = FirestoreChatMessageHistory(session_id=SESSION_ID, collection=COLLECTION_NAME, client=client)
    
    summary_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    memory = ConversationSummaryBufferMemory(llm=summary_llm, chat_memory=chat_memory, return_messages=True, max_token_limit=1024, memory_key="chat_history")

    print("开始与贤狼赫萝聊天吧! 输入 ’退出‘，‘quit’ 或 ’exit‘ 以结束对话。")
    while True:
        query = input("您: ")
        if query.lower() in ["退出", "quit", "exit"]:
            break
        response = agent_executor.invoke({"input": query, "chat_history": memory.chat_memory.messages})
        print("赫萝: " + response["output"])
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response["output"])

def main():
    load_dotenv()
    chat()

if __name__ == "__main__":
    main()