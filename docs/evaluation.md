# 预约 Agent 使用与评测

## 售后对话

向 `POST /chat/stream` 发送 JSON。模型读取工具结果后返回咨询答复或待确认提案。后续对话沿用同一个会话和浏览器身份。

```json
{"message":"R100 路由器 PON 灯变红，请帮我排查故障。","scenario":"after_sales"}
```

## GPU 模型选型

RTX 4070 Laptop 8 GB，llama.cpp Vulkan，统一 Q8_0、同一 51 条冻结题、单并发、种子 42、温度 0、生成上限 512 tokens。预热后每组正式评测一次。正确性为逐样本任务分均值，有效通过要求协议通过且任务分至少 0.95。

| 模型 | 任务正确性均分 | 严格有效通过 | 平均时延 | 平均解码吞吐 |
|---|---:|---:|---:|---:|
| Qwen3 0.6B | 38.37% | 0/51 | 0.90 s | 131.53 tok/s |
| Qwen3 1.7B | 68.89% | 9/51 | 1.06 s | 101.37 tok/s |
| Qwen3 4B | 80.01% | 28/51 | 2.15 s | 49.46 tok/s |

[逐题分数和计时](gpu-evaluation.json)。1.7B 较 4B 的平均时延降幅为 50.92%，解码吞吐增幅为 104.93%，任务正确性均分低 11.12 个百分点。

## LoRA 后训练

本地规则生成 500 条合成 Raw，派生 500 条 SFT 与 150 对 DPO。SFT 用 450 条训练，DPO 用 135 对训练。基于同一 Qwen3-1.7B，在 RTX 4070 Laptop 上用 BF16 LoRA 分别训练 1 轮。对同一 51 条冻结题评测时，Base、SFT、DPO 的任务正确性均分为 68.89%、86.89%、87.97%，严格有效通过为 9/51、19/51、21/51。该对照衡量冻结集表现。

## CI

[GitHub Actions 工作流](https://github.com/dojalance/smart-appointment-agent/actions/workflows/agent-evaluation.yml)覆盖门禁评分器和预约服务契约。本地同范围回归测试 33 项通过。
