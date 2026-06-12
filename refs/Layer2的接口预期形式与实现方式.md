# Layer 2 — 场景顾问 Agent · 接口与实现

## 1. 定位

有状态的多轮对话 Agent。理解用户的出行场景 → 拆成评估维度 → 并发调用 Layer 1 → 综合成结构化建议。

**只管"用户关心什么、该怎么回答"。不管怎么检索怎么融合。**

---

## 2. 接口

### 主接口

```python
def advise(
    user_input: str,          # 用户自然语言，如 "下周出差三天要开视频会议，合适吗？"
    session_id: str = "default"
) -> dict | AsyncIterator:
    """
    流式返回:
    {
        "scene": "商务出差",                    # 识别的场景类型
        "confidence": 0.92,                    # 场景置信度
        "dimensions": [                        # 评估的维度
            {"name": "网络质量", "importance": "●", "query": "WiFi视频会议稳吗？"},
            {"name": "安静程度", "importance": "●", "query": "隔音好不好？"},
            {"name": "退房效率", "importance": "●", "query": "退房快不快？"},
            {"name": "位置交通", "importance": "●", "query": "交通方便吗？"}
        ],
        "verdict": "适合出差，但需要选对房间",     # 一句话结论
        "reply": str,                           # 结构化 Markdown 回复（见下文格式）
        "sources": [                            # 聚合后的引用来源
            {"dimension": "安静", "content": "...", "ref_ids": [3,7,15]}
        ],
        "metrics": {
            "dimension_count": 4,
            "total_latency": 4.3
        }
    }
    """
```

### 回复格式

Layer 2 生成的 `reply` 遵循固定模板：

```markdown
好的，<场景名>，帮你逐项评估：

<维度1 emoji> <维度名> — <判断(好/风险/注意)>
<引用住客原话>「...」
→ <可操作建议>

<维度2 emoji> <维度名> — <判断>
...

📋 **结论：<一句话判断>**
<2-3 条可操作建议>
```

---

## 3. 内部流程

```
user_input
  │
  ▼
场景识别 ─── LLM few-shot 分类为 6 个场景之一，提取 specifics
  │
  ▼
维度映射 ─── 查矩阵 → 确定核心/次要维度 → 为每个维度生成自然语言 query
  │
  ▼
并行查询 ─── asyncio.gather( layer1.ask(q1), layer1.ask(q2), ... )
  │
  ▼
综合研判 ─── LLM 阅读各维度 Layer 1 返回 → 判好/坏/注意 → 给出总体结论
  │
  ▼
结构化回复 ── 套用模板输出
```

### 3.1 场景识别

用 LLM few-shot 分类。Prompt 示例：

```
你是酒店入住决策顾问。识别以下用户输入属于哪种出行场景：

场景选项：商务出差 / 亲子家庭 / 情侣蜜月 / 带长辈出行 / 朋友出游 / 独自旅行

用户输入：{user_input}

返回 JSON：{"scene": "...", "specifics": {...}, "confidence": 0.xx}
```

### 3.2 维度映射

核心资产：场景-维度矩阵。

| 维度 | 💼商务 | 🍼亲子 | 💑蜜月 | 🧓长辈 | 👥朋友 | 🎒独自 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 网络 | ● | — | — | — | ○ | ○ |
| 安静 | ● | ● | ● | ● | — | ○ |
| 退房效率 | ● | — | — | — | — | — |
| 位置交通 | ● | — | — | — | ● | ● |
| 安全 | — | ● | — | — | — | ● |
| 儿童设施 | — | ● | — | — | — | — |
| 景观/私密 | — | — | ● | — | — | — |
| 餐饮体验 | — | — | ● | — | ○ | — |
| 无障碍 | — | — | — | ● | — | — |
| 服务响应 | — | — | — | ● | — | — |
| 性价比 | ○ | — | — | — | ● | ● |
| 周边配套 | — | ○ | — | ○ | ● | — |

● 核心维度（必查）　○ 次要维度（LLM 根据 specifics 决定）

每个维度有预设的 query 模板，LLM 填入 specifics 后生成最终 query。例如：

```
安静程度:
  商务: "出差需要安静休息，房间隔音怎么样？有没有噪音投诉？"
  亲子: "宝宝需要午睡，房间隔音好不好？电梯声音会不会吵？"
  蜜月: "房间私密性怎么样？能不能听到走廊或隔壁的声音？"
```

### 3.3 并行查询

```python
import asyncio

async def query_dimensions(layer1, dimension_queries):
    tasks = [layer1.ask(q, strategy="hybrid") for q in dimension_queries]
    return await asyncio.gather(*tasks)
```

带场景特定的 filters：

```python
# 如果用户提到具体时间，注入 date_range
layer1.ask("安静吗？", filters={"date_range": ["2024-09", "2024-10"]})
# 如果用户提到房型，注入 room_type
layer1.ask("设施怎么样？", filters={"room_type": "花园大床房"})
```

### 3.4 综合研判

将各维度的 Layer 1 返回汇集，发给 LLM：

```
你是酒店入住决策顾问。用户场景：{scene}。以下是各维度的住客反馈：

网络质量：{q1_result}
安静程度：{q2_result}
退房效率：{q3_result}
位置交通：{q4_result}

请综合判断：
1. 每个维度是"好"、"风险"还是"注意"（一句话）
2. 每个维度给出 1 条可操作建议
3. 整体给出"推荐/谨慎推荐/不推荐"的结论及理由
```

### 3.5 结构化回复

LLM 输出按模板格式：

```markdown
好的，**商务出差场景**。帮你逐项评估：

📶 **网络** — 基本没问题，但有小坑
大部分住客说「WiFi速度快」，但有明确提到视频会议时「断断续续」「掉线」。
→ 入住时问前台哪个房间信号最好，或者自带移动热点。

🔇 **安静 — 这是最大的风险点**
187条噪音投诉集中在……
→ 务必选朝花园高层，远离电梯，最好走廊尽头。

📋 **结论：适合出差，但有三个前提**
① 选朝花园高层、走廊尽头的房间
② 确认房间 WiFi 信号，备好移动热点
③ 提前一晚结账，避开早高峰退房
```

---

## 4. 实现方式

### 4.1 文件结构

```
advisor/
├── agent.py           # ScenarioAgent 主控：advise() 接口
├── scenes.py           # 场景-维度矩阵 + query 模板 + 回复模板
├── recognizer.py       # 场景识别：LLM few-shot → {scene, specifics, confidence}
├── synthesizer.py      # 综合研判 + 结构化回复生成
└── memory.py           # 多轮对话记忆（InMemorySaver / 摘要压缩）
```

### 4.2 多轮记忆

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# 用户追问 "那刚才说的花园房呢？"
# → 从 checkpointer 获取历史 → 知道场景仍是"商务出差" → 聚焦花园房维度
# 对话过长时用 SummarizationMiddleware 压缩早期内容
```

### 4.3 关键依赖

```
langchain, langchain-openai, langgraph
openai (DashScope compatible)
pyyaml
```

### 4.4 配置项（config.yaml）

```yaml
agent:
  max_parallel_queries: 4     # 并发查询上限
  memory_max_tokens: 4000     # 记忆窗口
  streaming: true             # 流式输出
```

### 4.5 CLI 入口

```python
# cli.py
from advisor.agent import ScenarioAgent
from engine.client import ReviewQAClient

layer1 = ReviewQAClient("config.yaml")
layer2 = ScenarioAgent(layer1, "config.yaml")

# 交互循环
while True:
    user_input = input("> ")
    async for event in layer2.advise(user_input, session_id="main"):
        if event["type"] == "text":
            print(event["content"], end="", flush=True)
```

---

## 5. 设计原则

- **Layer 1 是无状态的工具**：Layer 2 不依赖 Layer 1 的任何内部状态
- **矩阵可扩展**：新增场景只需在 `scenes.py` 的矩阵加一列 + query 模板
- **策略透明**：`advise()` 返回中带 `dimensions` 和 `metrics`，用户可看到系统关注了什么维度
- **并行优先**：各维度互不依赖，并发查询保证延迟约等于最慢的那路而非各路之和
