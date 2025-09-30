# 《AI Agent 实战营》配套代码

本目录包含《AI Agent 实战营》的所有配套示例代码。每个项目都是可独立运行的完整示例。

## 📚 项目结构

所有项目按周次组织，涵盖了从基础概念到高级技术的完整学习路径。

## 🚀 Week 1 - Agent 基础

### 1. learning-from-experience - 强化学习 vs LLM 对比
`week1/learning-from-experience/`

对比传统强化学习（Q-learning）与基于 LLM 的上下文学习，复现 Shunyu Yao 的 "The Second Half" 博文中的关键洞察。通过寻宝游戏展示 LLM 如何以 250-400 倍的样本效率超越传统 RL。

**核心概念**：强化学习、上下文学习、样本效率、先验知识

### 2. web-search-agent - Kimi K2 模型即 Agent
`week1/web-search-agent/`

实现具备基础深度搜索能力的 Agent，能够进行多轮搜索和信息整合。

**核心概念**：网络搜索、模型原生 Agent

### 3. search-codegen - GPT-5 原生工具集成
`week1/search-codegen/`

构建能够基础深度搜索能力和代码沙盒能力的 Agent，综合利用网络搜索、代码执行等工具实现复杂分析。

**核心概念**：网络搜索、代码生成、模型原生 Agent

### 4. context - 上下文消融研究 
`week1/context/`

通过系统性的消融实验展示 Agent 上下文各个组件的重要性。支持多种 LLM 提供商（SiliconFlow Qwen、字节 Doubao、月之暗面 Kimi），配置不同的上下文模式观察 Agent 行为变化。

**核心概念**：上下文管理、工具调用、ReAct 循环、消融研究

## 🎯 Week 2 - 上下文工程与优化

### 1. local_llm_serving - 本地 LLM 部署与工具调用
`week2/local_llm_serving/`

跨平台的本地 LLM 部署方案，自动选择最佳后端（vLLM 或 Ollama）。展示即使 0.6B 的小模型也能通过良好的系统设计实现出色的工具调用能力。支持流式响应，实时显示思考过程。

**核心概念**：模型部署、Chat Template、流式处理、工具调用

### 2. attention_visualization - 注意力机制可视化
`week2/attention_visualization/`

可视化 LLM 的完整输入输出 token 序列和注意力权重分布，深入理解模型如何处理上下文、进行推理和调用工具。

**核心概念**：注意力机制、token 分析、推理过程可视化

### 3. kv-cache - KV Cache 友好的上下文设计
`week2/kv-cache/`

探索不同上下文管理模式对 KV Cache 的影响，演示常见的错误模式如何破坏缓存效率。通过实验展示正确的上下文设计如何显著降低延迟和成本。

**核心概念**：KV Cache、上下文优化、性能调优

### 4. context-compression - 上下文压缩策略
`week2/context-compression/`

实现并对比多种上下文压缩策略，包括摘要、关键信息提取、语义压缩等。在保持 Agent 能力的同时减少 token 使用量。

**核心概念**：上下文压缩、token 优化、信息密度

### 5. prompt-engineering - 提示工程消融研究
`week2/prompt-engineering/`

扩展 Tau-Bench 框架，通过系统性的消融实验量化不同提示工程因素对 Agent 性能的影响。展示语气风格、指令组织、工具描述等因素如何影响任务完成率。

**核心概念**：提示工程、消融研究、性能基准测试

### 6. system-hint - 系统提示优化
`week2/system-hint/`

研究系统提示（System Hint）对 Agent 行为的影响，探索如何通过优化系统提示提升性能。

**核心概念**：系统提示、行为引导、提示优化

### 7. user-memory-evaluation - 用户记忆评估框架
`week2/user-memory-evaluation/`

系统化评估用户记忆系统的准确性、相关性和有效性，包含多种测试场景和评估指标。

**核心概念**：评估框架、测试用例、性能度量

### 8. user-memory - 用户记忆系统
`week2/user-memory/`

构建长期用户记忆系统，让 Agent 能够记住用户偏好和历史交互，提供个性化服务。

**核心概念**：长期记忆、个性化、用户建模

### 9. log-sanitization - 日志脱敏处理
`week2/log-sanitization/`

实现智能的日志脱敏系统，在保留调试信息的同时保护敏感数据。

**核心概念**：隐私保护、日志处理、数据安全

## 📚 Week 3 - 知识库与学习机制

### 1. dense-embedding - 稠密嵌入向量检索服务
`week3/dense-embedding/`

构建向量相似性搜索服务，对比研究 ANNOY（基于树）和 HNSW（基于图）两种近似最近邻索引算法。展示不同索引策略在性能、内存占用和更新能力上的权衡。

**核心概念**：稠密嵌入、向量检索、ANN 算法、语义搜索

### 2. sparse-embedding - 稀疏检索引擎
`week3/sparse-embedding/`

从零实现基于 BM25 算法的稀疏向量搜索引擎，通过丰富的日志和可视化接口展示搜索引擎的内部工作机制，理解词频权重计算和倒排索引原理。

**核心概念**：稀疏嵌入、BM25、TF-IDF、精确匹配

### 3. retrieval-pipeline - 混合检索流水线
`week3/retrieval-pipeline/`

构建完整的检索流水线，结合稠密检索、稀疏检索和神经重排序。通过精心设计的测试用例，系统性展示混合检索在不同场景下的优势互补效果。

**核心概念**：混合检索、神经重排序、跨编码器、检索融合

### 4. multimodal-agent - 多模态信息提取
`week3/multimodal-agent/`

对比三种多模态处理策略：原生多模态处理、提取为文本、工具化分析。通过统一框架下的消融研究，揭示不同技术路径在保真度、成本和灵活性上的权衡。

**核心概念**：多模态、视觉理解、OCR、端到端处理

### 5. structured-index - 结构化索引
`week3/structured-index/`

实现并对比 RAPTOR（递归抽象树）和 GraphRAG（知识图谱）两种先进索引策略。通过索引技术手册演示如何构建反映知识内在层次和关联的结构化索引。

**核心概念**：RAPTOR、GraphRAG、层次摘要、知识图谱

### 6. agentic-rag - 智能体 RAG
`week3/agentic-rag/`

对比传统 Non-Agentic RAG 与 Agentic RAG 的性能差异。展示 Agent 如何通过 ReAct 模式主导迭代式信息检索，在处理复杂司法问答时显著提升答案质量。

**核心概念**：Agentic RAG、ReAct 循环、迭代检索、主动探索

### 7. agentic-rag-for-user-memory - 利用 Agentic RAG 构建用户记忆
`week3/agentic-rag-for-user-memory/`

将 Agentic RAG 框架应用于管理用户对话历史，通过多轮迭代搜索能力处理跨会话的记忆检索，实现基础回忆和多会话检索能力。

**核心概念**：用户记忆、对话历史索引、跨会话检索

### 8. contextual-retrieval - 上下文感知检索
`week3/contextual-retrieval/`

实现 Anthropic 提出的上下文感知检索技术，通过为文本块生成包含核心上下文的前缀摘要，解决传统分块方法的上下文丢失问题，将检索失败率降低 49-67%。

**核心概念**：上下文增强、前缀生成、语义锚定、检索优化

### 9. contextual-retrieval-for-user-memory - 上下文感知的用户记忆系统
`week3/contextual-retrieval-for-user-memory/`

将上下文感知检索技术应用于用户记忆构建，结合 Advanced JSON Cards 与上下文感知 RAG，形成双层记忆结构，实现更高层次的主动服务能力。

**核心概念**：双层记忆、结构化事实、上下文检索、主动服务

### 10. structured-knowledge-extraction - 结构化知识提取
`week3/structured-knowledge-extraction/`

从海量司法判例数据集中提取隐性知识，通过因子分析和重要性建模，构建判决经验模型。展示如何将数据中的隐性模式转化为 Agent 可用的结构化决策逻辑。

**核心概念**：知识发现、因子分析、数据驱动、判决建模

### 11. gaia-experience - 从成功经验中学习
`week3/gaia-experience/`

基于 AWorld 框架和 GAIA 基准测试，实现完整的"学习-应用"闭环。Agent 自动总结成功的任务轨迹为结构化经验，并在新任务中检索应用，实现自我进化。

**核心概念**：经验学习、策略摘要、轨迹总结、自我进化

### 12. browser-use-rpa - 工作流录制与回放
`week3/browser-use-rpa/`

实现浏览器自动化的工作流录制系统，将重复性操作序列自动封装为参数化工具。通过从昂贵的 LLM 推理切换到精确的自动化执行，实现 3-5 倍速度提升。

**核心概念**：工作流录制、RPA、工具生成、外部化学习

## 🛠️ Week 4 - 工具生态与系统集成

### 1. perception-tools - 感知工具 MCP 服务器
`week4/perception-tools/`

构建全面的感知工具集，提供网络搜索、多模态理解、文件系统操作和公共数据源访问能力。大部分功能基于免费开放 API（DuckDuckGo、Open-Meteo、Yahoo Finance、OpenStreetMap 等），无需 API 密钥即可使用。

**核心概念**：MCP 协议、多模态解析、公共数据源、文档理解、地理信息服务

### 2. execution-tools - 执行工具 MCP 服务器
`week4/execution-tools/`

实现具备安全机制的执行工具集，包括文件操作、代码解释器、虚拟终端和外部系统集成。通过 LLM 二次审批机制防止危险操作，自动摘要复杂输出，并对代码进行语法验证。

**核心概念**：MCP 协议、执行安全、LLM 审批、结果摘要、自动验证

### 3. collaboration-tools - 协作工具 MCP 服务器
`week4/collaboration-tools/`

提供全面的协作能力，包括浏览器自动化（browser-use 框架）、人机协同（Human-in-the-Loop）、多渠道通知（Email、Telegram、Slack、Discord）和定时器管理。支持敏感操作的管理员审批和定时任务调度。

**核心概念**：MCP 协议、浏览器自动化、HITL 模式、多渠道通知、定时任务

### 4. agent-with-event-trigger - 事件触发型 Agent 与 MCP 集成
`week4/agent-with-event-trigger/`

基于 FastAPI 构建的现代化事件驱动 Agent，默认集成前三个 MCP 服务器的所有工具。采用原生异步架构实现清晰的 MCP 工具加载，通过 HTTP API 接收多源事件（Web、即时消息、GitHub、定时器等）。提供自动 API 文档（Swagger UI）和后台监控能力。

**核心概念**：FastAPI、原生异步、MCP 集成、事件驱动、自动 API 文档、工具编排

## 📖 学习建议

1. **按顺序学习**：Week 1 建立基础概念，Week 2 深入工程实践，Week 3 探索知识库与学习机制，Week 4 掌握工具生态与系统集成
2. **动手实践**：每个项目都设计为可独立运行，建议亲自运行并修改代码
3. **结合书籍**：配合《AI Agent 实战营》配套电子书相应章节阅读，理解理论与实践的结合
4. **实验对比**：多个项目包含消融研究和对比实验，通过对比加深理解
5. **渐进学习**：Week 4 的 MCP 服务器项目展示了标准化工具协议，建议先熟悉 MCP 基础概念再深入学习各个工具模块

## 🔑 API 密钥

建议大家申请几个平台的 API key，方便学习：
- **Kimi**: https://platform.moonshot.cn/
- **Siliconflow**: https://siliconflow.cn/ 上面有各种开源模型，包括 DeepSeek、Qwen 等
- **火山引擎**: https://www.volcengine.com/product/ark 上面有字节的闭源模型（豆包），国内访问延迟比较低
- **OpenRouter**: https://openrouter.ai/ 可以从国内直接访问海外的各种闭源和开源模型，包括 Gemini 2.5 Pro、Claude 4 Sonnet、OpenAI GPT-5 等（官方 API 需要海外 IP 和支付方式，OpenAI 还需要海外身份实名认证，注册比较麻烦）

模型选型可以参考： https://01.me/2025/07/llm-api-setup/

## 🤝 贡献

欢迎通过 Pull Request 贡献代码改进、bug 修复或新的示例项目。

## 📄 许可证

本项目代码仅供学习参考，具体许可证信息请查看各子项目。
