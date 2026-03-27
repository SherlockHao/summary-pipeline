# 开发 Agent 团队编排

## 团队成员与职责

基于 PRD 的模块划分，团队分为 **6 个 Agent 角色**，每个角色负责独立模块，可并行开发。

---

### Agent 1: 数据工程师 — `data-engineer`
**负责模块：** 数据输入层 + 预处理层（PRD §1.1.1 - §1.1.2）

| 任务 | 产出 |
|:---|:---|
| DingTalk A1 / AI听记 JSON 适配器 | `models/asr_result.py` |
| ASR + Speaker Diarization 数据对齐 | 时间-说话人-内容三元组 |
| 数据清洗（置信度过滤、时间戳修正） | 清洗 pipeline |
| 话语合并（3-5秒间隔阈值） | 合并逻辑 |
| 特征计算（时长、轮次、关键词密度） | 特征提取模块 |

**关键文件：** `models/asr_result.py`, `core/preprocessor.py`

---

### Agent 2: 策略引擎师 — `strategy-engineer`
**负责模块：** 智能分层 + 分片策略引擎（PRD §1.1.3, §1.2.2）

| 任务 | 产出 |
|:---|:---|
| 时间窗口分片（45min默认，30-60自适应） | 分片策略插件 |
| 主题边界分片（Sentence-BERT语义相似度） | 主题检测 |
| 关键人优先分片（保护区机制） | 保护区逻辑 |
| 三策略动态组合与优先级仲裁 | `core/chunking_engine.py` |

**关键文件：** `core/chunking_engine.py`, `config/chunking_strategy.yaml`

---

### Agent 3: 评分算法师 — `scoring-engineer`
**负责模块：** 重要性评估模型（PRD §1.1.3, §1.2.4, §4）

| 任务 | 产出 |
|:---|:---|
| 显式特征计算（关键人、时间、结构） | 规则评分引擎 |
| 隐式特征提取（LLM轻量级调用） | 语义评分 |
| 多维度加权融合（α=0.6/0.7切换） | 融合算法 |
| 短对话保护机制（P0最低80分） | 保护逻辑 |
| 连续性保护（重要会话簇） | 簇合并逻辑 |

**关键文件：** `core/importance_evaluator.py`, `models/scoring.py`

---

### Agent 4: 摘要生成师 — `summarizer-engineer`
**负责模块：** 多级摘要生成 + 提示词工程（PRD §1.1.4, §5）

| 任务 | 产出 |
|:---|:---|
| 片段级摘要（非思考模式，结构化JSON） | L1 summarizer |
| 时段级摘要（思考模式，去重+关联） | L2 summarizer |
| 全天级摘要（思考模式，全局视角） | L3 summarizer |
| 提示词模板设计（角色定义+反幻觉） | prompt templates |
| 摘要融合与去重（Sentence-BERT > 0.85） | 融合模块 |

**关键文件：** `core/summarizer.py`, `models/summary.py`, `prompts/`

---

### Agent 5: 模型接口师 — `api-engineer`
**负责模块：** Qwen3-Max 调用封装 + Token预算控制（PRD §1.2.3, §2）

| 任务 | 产出 |
|:---|:---|
| Qwen3-Max API 客户端封装 | `api/qwen_client.py` |
| Token 精确计算（官方 tokenizer） | token counter |
| 动态分片触发（>17K tokens） | 预算控制器 |
| 重试 + 降级 + 流式处理 | `api/retry_handler.py` |
| 成本监控与预算告警 | cost tracker |

**关键文件：** `api/qwen_client.py`, `api/retry_handler.py`, `api/streaming.py`

---

### Agent 6: 配置与测试师 — `config-test-engineer`
**负责模块：** 配置管理 + 测试数据集（PRD §1.2.1, §3, §6）

| 任务 | 产出 |
|:---|:---|
| 关键人 YAML/JSON 配置 Schema | `config/key_people.yaml` |
| 热更新机制（文件监听+原子替换） | config loader |
| 版本控制与回滚 | version manager |
| 模拟数据生成器（4种场景覆盖） | test data generator |
| 基准测试框架（ROUGE-L, BERTScore） | benchmark suite |

**关键文件：** `config/`, `tests/`

---

## 依赖关系与开发顺序

```
Phase 1 (并行):
  Agent 1 [数据工程师]  ──→ 数据模型 & 预处理
  Agent 5 [模型接口师]  ──→ API封装 & Token控制
  Agent 6 [配置测试师]  ──→ 配置Schema & 测试数据

Phase 2 (依赖Phase 1):
  Agent 2 [策略引擎师]  ──→ 分片引擎 (依赖: Agent 1的数据模型)
  Agent 3 [评分算法师]  ──→ 评分系统 (依赖: Agent 1的特征, Agent 6的配置)

Phase 3 (依赖Phase 1+2):
  Agent 4 [摘要生成师]  ──→ 三级摘要 (依赖: Agent 2分片, Agent 3评分, Agent 5 API)
```

## 当前状态

| Agent | 角色 | 状态 |
|:---|:---|:---|
| Agent 1 | 数据工程师 | 🟡 STANDBY |
| Agent 2 | 策略引擎师 | 🟡 STANDBY |
| Agent 3 | 评分算法师 | 🟡 STANDBY |
| Agent 4 | 摘要生成师 | 🟡 STANDBY |
| Agent 5 | 模型接口师 | 🟡 STANDBY |
| Agent 6 | 配置与测试师 | 🟡 STANDBY |
