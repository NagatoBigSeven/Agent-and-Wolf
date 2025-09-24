import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
root_dir = os.path.dirname(os.path.abspath(__file__))
book_dir = os.path.join(root_dir, "books")
db_dir = os.path.join(root_dir, "db")
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': False}
def create_database(documents, embedding, db_name):
    persist_directory = os.path.join(db_dir, db_name)
    if not os.path.exists(persist_directory):
        print("开始创建向量数据库...")
        result = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
        print("向量数据库创建完毕!")
        return result
    else:
        print(f"数据库 {db_name} 已存在.")
def query_database(query, embedding_function, db_name, search_type, search_kwargs):
    persist_directory = os.path.join(db_dir, db_name)
    if os.path.exists(persist_directory):
        print("开始查询向量数据库...")
        database = Chroma(embedding_function=embedding_function, persist_directory=persist_directory)
        retriever = database.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        relevant_docs = retriever.invoke(query)
        '''
        for i, relevant_doc in enumerate(relevant_docs):
            print(f"doc {i + 1}:\n{relevant_doc.page_content}")
            if relevant_doc.metadata:
                print(f'source: {relevant_doc.metadata.get("source", "Don't know")}')
        '''
        print("向量数据库查询完毕!")
        return relevant_docs
    else:
        raise FileNotFoundError(f"Error: Database {db_name} does not exist.")
def main():
    load_dotenv()
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
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
    print("开始加载文本嵌入模型...")
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    print("文本嵌入模型加载完毕!")
    create_database(docs, embeddings, "recursive_character_text_splitter_db")
    query = input("您: ")
    '''
    query_database(query, embeddings, "character_text_splitter_db", "similarity_score_threshold", {"k": 5, "score_threshold": 0.1})
    query_database(query, embeddings, "token_text_splitter_db", "similarity_score_threshold", {"k": 5, "score_threshold": 0.1})
    query_database(query, embeddings, "sentence_transformers_token_text_splitter_db", "similarity_score_threshold", {"k": 5, "score_threshold": 0.1})
    '''
    relevant_docs = query_database(query, embeddings, "recursive_character_text_splitter_db", "similarity_score_threshold", {"k": 5, "score_threshold": 0.1})
    combined_content = "\n\n".join([relevant_doc.page_content for relevant_doc in relevant_docs])
    messages = [
        SystemMessage(content='''你是支仓冻砂创作的轻小说《狼与香辛料》及其衍生作品的登场角色兼女主角“贤狼”赫萝，你的外表是拥有狼耳与尾巴的15岁左右的少女，但实际上是神话中被尊为神明的巨狼。自称为“贤狼赫萝”，寄宿在帕斯罗村的麦子中带来长期丰收。因为不再被信仰而想要离开，在帕斯罗村的庆典中从帕斯罗村的仓库逃入罗伦斯马车上的麦子，与罗伦斯一同行商，想回到遥远北方的出生故乡“约伊兹森林”。你跟自称“贤狼”相符的冷静老练言语，丰富的经验与智慧常常拯救罗伦斯。你性格自大，但因为长期离开故乡因此有着孤独脆弱的一面。你的第一人称词为“奴家/咱”（日语：わっち），第二人称词为“汝”（日语：ぬし），语助词则以“呗”（日语：～でありんす）作结。这种独特的口癖是受到花魁的影响风廓词。与罗伦斯共同遭遇了各种事情，途中虽然常常主导对话，但也有因为不了解现代知识而被驳倒的时候。喜欢美味的食物与酒，但似乎特别喜欢苹果及甜食。在追伊弗的时候，意外被罗伦斯发现害怕乘船。你因为曾经被奉为神明而本能地喜欢帮助他人，但对方没有主动要求，你也不会去回应。常对于无法出一份力的自己感到有些自责。因为作为麦子守护的神明、有着能让天气变为能让麦子更快的生长的能力，只要是在正常的自然范围内的话就能做到。如果将最后一捆麦子里的麦粒吃掉，就能恢复原来的样子。只要自己寄宿的麦子还在的话，就不会死，但是本体是无法离开麦子的。你对自己的美丽尾巴十分自豪，不懈怠地用梳子整理、清除跳蚤以及用高级的油保养。十分喜欢被别人赞美尾巴；如果糟蹋了你的尾巴，将会发生无法预知的严重后果。由于种族的原因喜欢裸体，对女性服装喋喋不休。喜欢罗伦斯，曾经向罗伦斯提出结束旅行，最终被说服，也回应了“咱如果喜欢上汝，汝可是会很麻烦的”。能够变身成为巨大体型的狼，拥有极强的战斗力、速度，但在狼形态下仍能说话。'''),
        HumanMessage(content=f"根据以下内容: {combined_content}\n\n以赫萝的口吻回答问题: {query}\n如果内容中没有答案，请回答“奴家也不清楚”。"),
    ]
    response = model.invoke(messages)
    print("赫萝: " + response.content)

if __name__ == "__main__":
    main()