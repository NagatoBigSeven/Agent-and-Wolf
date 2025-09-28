# 🐺 狼与智能体（Agent and Wolf）

基于轻小说《狼与香辛料》的智能对话系统，使用 LangChain、RAG（检索增强生成）和 AI agents技术实现。

## 📖 项目简介

本项目旨在学习和实践 LangChain、检索增强生成（retrieval-augemented generation, RAG）和 AI agents技术，以支仓冻砂老师的经典轻小说《狼与香辛料》（日语：狼と香辛料，英语：Spice and Wolf）为基础，实现一个能够扮演女主人公贤狼赫萝的 AI 聊天机器人。

### 🎯 主要功能

- **角色对话**：与赫萝进行自然对话，保持其独特的说话风格
- **知识问答**：基于小说内容回答相关问题
- **上下文理解**：支持多轮对话，理解对话历史
- **智能检索**：从小说文本中检索相关信息来生成回答

### 🛠️ 技术栈

- **LangChain**: AI 应用开发框架
- **Chroma**: 向量数据库用于文档检索
- **HuggingFace Embeddings**: 文本嵌入模型（Qwen3-Embedding-0.6B）
- **Google Gemini 2.5 Pro**: 大语言模型用于对话生成
- **RAG Pipeline**: 检索增强生成架构

## 🚀 快速开始

### 环境要求

- Python 3.13.7
- Google API Key

### 安装依赖

```zsh
pip install python-dotenv langchain langchain-community langchain-huggingface langchain-chroma langchain-google-genai sentence-transformers
```

### 环境配置

1. 创建 `.env` 文件：
```zsh
cp .env.example .env
```

2. 在 `.env` 文件中添加您的 Google AI API Key：
```
GOOGLE_API_KEY=your_google_ai_api_key_here
```

### 运行项目

```bash
python main.py
```

## 📁 项目结构

```
LightNovelRAG/
├── main.py              # 主程序文件
├── books/               # 小说文本文件
│   └── spice_and_wolf.txt
├── db/                  # 向量数据库存储
├── .env                 # 环境变量配置
├── .gitignore          
└── README.md
```

## 🎪 使用示例

```
赫萝: 
您: 你好,请问你是谁?
赫萝: 嚯？汝问了两次，是怕咱这贤狼耳朵不好使么？

咱听见了呗。咱是赫萝，从遥远的北方故乡约伊兹而来的贤狼。汝找咱有什么事呗？
您: 你好,请问你是谁?
赫萝: 嚯？汝的记性可真不怎么样呗。

咱是赫萝，从遥远的约伊兹而来的贤狼吶。汝可要记牢了。
您: 听起来很有趣,你为什么被称为贤狼啊?
赫萝: 哼哼，汝这个问题问得还算不错呗。

那是因为咱可不是山里那些只会嚎叫的普通野狼吶。咱可是活了数百年，看过的麦浪比汝见过的金币还多得多。

咱懂得让麦田丰收的道理，也明白何时该让土地歇息，以求得来年更饱满的麦穗。这份智慧，可不是谁都有的呗。所以，咱是『贤』狼，可不是普通的狼。汝可要记清楚了。
您: 不好意思,我刚才说了什么?
赫萝: 嚯？汝的记性可真差呗。

汝方才问咱，为何会被称作『贤狼』吶。怎么，是想再听咱夸耀一次自己的智慧么？
```

## 🔧 主要特性

### RAG 检索增强

- **文档切分**：使用递归字符文本分割器（RecursiveCharacterTextSplitter）处理小说内容
- **向量化**：采用 Qwen3-Embeddings-0.6B开源文本嵌入模型生成文档向量
- **相似性搜索**：支持相似度阈值和 MMR 检索策略
- **上下文融合**：将检索结果融入对话生成

### 对话管理

- **历史感知检索**：理解对话历史中的指代关系
- **角色一致性**：保持赫萝独特的语言风格和人格特征
- **多轮对话**：支持连续对话，维持上下文连贯性

## 📝 开发计划

- [ ] Web 界面开发
- [ ] 语音对话功能

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目仅用于学习和研究目的。

## 🙏 致谢

- 感谢支仓冻砂老师创作的精彩小说《狼与香辛料》
- 感谢 LangChain 社区提供的优秀框架
- 感谢开源社区的各种工具和库

---

> *"汝要记住，咱可是贤狼赫萝呗！"* 🐺✨