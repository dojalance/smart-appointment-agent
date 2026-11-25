# 企业售后智能客服与预约 Agent

基于 FastAPI、LangChain 和 SQLite 的售后助手。通过 ReAct 调用产品咨询、预约准备和行为分析工具，将知识库里的故障信息和技能要求用于工程师匹配。

## 功能

- 流式聊天、维修知识咨询和档期查询。
- 生成预约提案，用户确认后写入。支持改约、取消和本地站内回访。
- SQLite 事务校验预约冲突和请求键，避免相同请求重复写入。
- 预约历史按用户隔离。支持按需召回偏好、近期上下文和抽取式摘要。
- 可通过 MCP 接入外部知识库，保留检索引用。

## 启动

需要 Python 3.12 和可用的模型服务。配置模板为 `.env.example`，复制为 `.env` 后填写 LLM 和 Embedding 的地址、模型及凭据。

```bash
python -m venv .venv
# 激活虚拟环境后运行
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

浏览器访问 `http://127.0.0.1:8001/`。售后对话可向 `POST /chat/stream` 发送 `scenario: "after_sales"`，请求示例见 [使用与评测](docs/evaluation.md)。配置 `AFTER_SALES_RAG_PYTHON` 和 `AFTER_SALES_RAG_ROOT` 可接入本地 MCP 服务。

## 评测

- [GPU 模型选型与 LoRA 训练结果](docs/evaluation.md)
- [GitHub Actions 预约与权限回归](https://github.com/dojalance/smart-appointment-agent/actions/workflows/agent-evaluation.yml)
