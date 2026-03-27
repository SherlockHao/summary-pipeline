# 架构文档第三方交叉审计汇总

> 审计日期：2026-03-27 | 6 Agent 闭环交叉审计

---

## 审计统计

| Agent | 自审模块 | 交叉审计 | 🔴阻塞 | 🟡需修正 | 🟢建议 |
|:--|:--|:--|:--|:--|:--|
| Agent 1 数据工程师 | 数据输入+预处理 | 分片引擎 | 3 | 8 | 3 |
| Agent 2 策略引擎师 | 分片引擎 | 评分模块 | 2 | 7 | 2 |
| Agent 3 评分算法师 | 评分模块 | 摘要生成 | 2 | 4 | 2 |
| Agent 4 摘要生成师 | 摘要生成 | API层 | 5 | 10 | 5 |
| Agent 5 模型接口师 | API层 | 配置管理 | 4 | 7 | 2 |
| Agent 6 配置测试师 | 配置管理 | 总架构+数据输入 | 3 | 7 | 3 |
| **合计（含重复）** | | | **19** | **43** | **17** |

---

## 🔴 阻塞问题去重汇总（编码前必须解决）

### BLOCK-1: Utterance 数据结构跨模块不兼容

**发现者**：Agent 1 (ISSUE-05), Agent 6 (ISSUE-05)

Module 1（数据输入）输出的 `Utterance` 与 Module 2（分片引擎）输入的 `Utterance` 完全不匹配：

| 维度 | Module 1 | Module 2 |
|:--|:--|:--|
| 时间单位 | 毫秒 `start_ms: int` | 秒 `start_time: float` |
| 唯一ID | 无 | `utterance_id: str` |
| token计数 | 无 | `token_count: int` |
| 关键人 | `key_person_match: KeyPersonMatch` | `key_person_level: KeyPersonLevel` |
| 可变性 | mutable `@dataclass` | frozen `@dataclass(frozen=True)` |

**修复方案**：在 `models/types.py` 中定义唯一的 `Utterance` 类，统一时间为秒/float，增加 utterance_id 和 token_count，Module 1 负责生产，Module 2 消费。

---

### BLOCK-2: Segment 类型未定义，Chunk→评分接口断裂

**发现者**：Agent 2 (ISSUE-2.1)

分片引擎输出 `Chunk`（含 sessions, utterances, total_tokens），评分模块输入 `Segment`（含 speakers, text, id）——但 `Segment` 从未在任何文档中定义。`Chunk` 没有 `text` 和 `speakers` 字段，评分模块无法获取所需数据。

**修复方案**：明确 `Chunk` 与 `Segment` 的关系（1:1 映射或 Chunk 包含多个 Segment），在共享模型中定义 `Segment`，或评分模块直接消费 `Chunk` 类型。

---

### BLOCK-3: ScoringResult 类型未定义，评分→摘要接口断裂

**发现者**：Agent 3 (ISSUE-2.1, 4.1)

评分模块返回 `tuple[list[Segment], list[ScoreBreakdown]]`，摘要模块期望 `ScoringResult` 对象——该类从未被定义。

**修复方案**：在 `models/scoring.py` 中定义 `ScoringResult` 数据类，包装 filtered_segments + all_breakdowns + mode 标记。

---

### BLOCK-4: 短日模式下 Top-K 筛选与 PRD 矛盾

**发现者**：Agent 3 (ISSUE-3.1)

PRD 明确说短日模式评分"不用于内容裁剪"，但评分模块对所有场景一律执行 Top-K 筛选（阈值60 + K=20）。如果摘要模块使用筛选后的结果，违反了"全量文本单次输入"原则。

**修复方案**：评分模块增加 `skip_filter: bool` 参数，短日模式下跳过 Top-K，仅返回排序+评分记录用于展示。

---

### BLOCK-5: 会话检测功能重复，数据流无法衔接

**发现者**：Agent 1 (ISSUE-03, ISSUE-06)

Module 1 预处理层 Step 1 做了会话检测（输出 `ProcessedSession` 列表），Module 2 的 SessionDetector 也做会话检测（期望输入扁平 `Utterance` 列表）。两者功能重叠且算法不同，Module 1 输出的容器结构与 Module 2 输入不匹配。

**修复方案**：会话检测只做一次。建议 Module 1 去掉 Step 1，输出扁平 `Utterance` 列表；会话检测统一由 Module 2 的 SessionDetector 执行（算法更完备，含 Jaccard 窗口和独白处理）。

---

### BLOCK-6: chat_completion 同步/异步 + 参数类型不匹配

**发现者**：Agent 4 (YI-06, YI-07)

摘要模块用 `await self.model_client.chat_completion(...)` 期望 async 方法，但 API 层定义为同步 `def`。摘要模块构建 messages 为 `list[dict]`，API 层期望 `list[ChatMessage]` dataclass。

**修复方案**：API 层 chat_completion 改为 `async def`，增加 `ModelClient` 协议/ABC 抽象层。messages 参数统一为 `list[ChatMessage]` 或 API 层内部做 dict→ChatMessage 转换。

---

### BLOCK-7: Token 预算参数差异 4-10 倍

**发现者**：Agent 4 (YI-15, YI-16), Agent 5 (ISSUE-4)

| 参数 | 摘要模块 | API层 | PRD |
|:--|:--|:--|:--|
| safety_margin | 20,000 | **2,000** | ≥20,000 |
| output_reserved | 32,768 | **8,000** | 30,000-32,768 |

API 层的 safety_margin 仅为 PRD 要求的 1/10，会导致 token 超限。

**修复方案**：API 层 safety_margin 修正为 20,000，output_reserved 按场景区分（日报级 32,768，片段级 4,000）。摘要模块应调用 TokenBudgetController 而非自行硬编码预算。

---

### BLOCK-8: model_params.yaml 两套不兼容 Schema

**发现者**：Agent 5 (ISSUE-1, 2.1)

API 层附录和配置模块各自定义了一套 model_params.yaml 的结构，字段名、层级、取值完全不同（如 retry 字段名 `base_delay` vs `backoff_base`，降级模型 `qwen-plus` vs `qwen3-plus`）。

**修复方案**：以配置模块的 Pydantic 模型为权威定义，API 层从 ConfigManager 读取所需字段。合并为一份统一 Schema。

---

### BLOCK-9: MatchResult vs KeyPersonMatch 类名和枚举值不一致

**发现者**：Agent 6 (ISSUE-03)

配置模块定义 `MatchResult`（L2 类型 `fuzzy`，未匹配 `unknown`），数据输入模块定义 `KeyPersonMatch`（L2 类型 `alias`，未匹配 `none`）。评分模块引用的是 `alias`。三方不一致。

**修复方案**：统一为 `MatchResult` + `MatchType` 枚举，L2 统一为 `fuzzy`（或 `alias`，二选一），在共享模型中定义。

---

### BLOCK-10: short_day_threshold 在不同文档中 80K vs 200K

**发现者**：Agent 1 (ISSUE-02), Agent 2 (ISSUE-1.1.1), Agent 6 (ISSUE-07)

PRD 5.0 明确短日 <80K，但分片引擎 token_budget=200K，总架构数据流图也标注 200K。三个 Agent 独立指出此矛盾。

**修复方案**：区分两个概念——(1) 摘要模式切换阈值 = 80K（由 SummaryOrchestrator 决策）；(2) 单 Chunk 最大容量 = 200K（分片引擎内部参数）。在文档中明确标注。

---

### BLOCK-11: QwenExhaustedError 在摘要模块中无处理

**发现者**：Agent 4 (YI-13)

API 层在主模型和降级模型均失败后抛出 `QwenExhaustedError`，但摘要模块无任何 try/except 处理，管道会直接崩溃。

**修复方案**：SummaryOrchestrator 捕获此异常，将任务推入 PRD 定义的"人工队列"，输出部分结果（已完成的时段摘要）+ 错误标记。

---

## 🟡 需修正问题分类汇总

### 接口/类型类（7项）
- 评分模块 `match_speaker_to_key_person()` 和 `LLMClient` 接口未定义
- 连续性保护伪代码对 float 调用 `.total_seconds()`（类型错误）
- ResponseParser 期望 str 但收到 ChatCompletionResponse 对象
- 配置模块 L3 LLM 客户端 `.call(prompt=...)` 与 QwenClient `.chat_completion(messages=...)` 不匹配
- 评分模块输出按分数降序，摘要模块需要按时间升序

### 参数不一致类（8项）
- 降级模型名称 `qwen-plus` vs `qwen3-plus`
- 尾部上下文摘要截断 400字 vs PRD 200-300字
- default_temperature 0.7 vs 0.3
- thinking_budget_default 16384 vs 16000
- L3 时间窗口 PRD 自身 30-60min vs 120min 矛盾
- 多信号融合阈值 theta=0.55 PRD 中缺失
- system_prompt 预算 2000 vs 1500

### 缺失/遗漏类（9项）
- `core/preprocessor.py` 在总架构模块清单中缺失
- ConfigManager 未加载 `chunking_strategy.yaml`
- Module 1 引用未定义的 `config/keywords.yaml`
- CostTracker 所需 cost 配置在配置模块中缺失
- timeout 配置在配置模块中缺失
- 冷启动阈值 0.25 在 KeyPersonMatcher 代码中未实际使用
- 摘要模块缺少 Batch API 集成设计
- RetryHandler 与 SummaryOrchestrator 集成方式未明确
- TokenBudgetController 从未被摘要模块调用

### 逻辑/设计类（5项）
- Module 1 `guess_scene` 函数无实现说明
- Module 2 `_subdivide_zone` 等关键函数仅有签名无伪代码
- embedding 模型选型和缓存策略未明确
- 短日模式 scoring 参数传入但未在 prompt 中使用
- 关键事实可能被 Top-K 过滤（长日模式）

---

## 修复优先级

### P0 — 编码前必须解决（11项阻塞）

| # | 问题 | 负责 Agent | 协作 |
|:--|:--|:--|:--|
| BLOCK-1 | 统一 Utterance 数据结构 | Agent 1 | Agent 2 |
| BLOCK-2 | 定义 Segment 类型 | Agent 3 | Agent 2 |
| BLOCK-3 | 定义 ScoringResult 类型 | Agent 3 | Agent 4 |
| BLOCK-4 | 评分模块增加 skip_filter | Agent 3 | — |
| BLOCK-5 | 会话检测去重 | Agent 1 | Agent 2 |
| BLOCK-6 | 统一 async + 参数类型 | Agent 5 | Agent 4 |
| BLOCK-7 | 修正 Token 预算参数 | Agent 5 | Agent 4 |
| BLOCK-8 | 合并 model_params Schema | Agent 5 | Agent 6 |
| BLOCK-9 | 统一 MatchResult 枚举 | Agent 6 | Agent 1 |
| BLOCK-10 | 区分 80K/200K 两个阈值 | Agent 2 | Agent 4 |
| BLOCK-11 | 补充 QwenExhaustedError 处理 | Agent 4 | Agent 5 |

### P1 — 编码早期解决（29项需修正）

接口/类型 7 项 + 参数不一致 8 项 + 缺失/遗漏 9 项 + 逻辑/设计 5 项

### P2 — 迭代完善（17项建议）

文档对齐、PRD 回写、边缘case 补充等
