# 数据输入层与预处理层 — 模块详细设计

> 对应 PRD §1.1.1（数据输入层）与 §1.1.2（预处理层）。
> 涉及文件：`models/asr_result.py`、`core/preprocessor.py`

---

## 一、数据输入层（models/asr_result.py）

### 1.1 核心数据模型

#### 1.1.1 Segment — 最小粒度片段

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class AlignmentQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass(frozen=True)
class Segment:
    """ASR + Speaker Diarization 对齐后的最小粒度片段。"""
    start_ms: int                    # 片段起始时间（毫秒，相对于 session_start）
    end_ms: int                      # 片段结束时间（毫秒）
    speaker_id: str                  # 说话人标识，如 "spk_001"
    text: str                        # ASR 识别文本
    asr_confidence: float            # ASR 识别置信度，0.0-1.0
    speaker_confidence: float        # 说话人归属置信度，0.0-1.0
    alignment_quality: AlignmentQuality  # ASR 与说话人分离的时间对齐质量
    # ---- 以下为可选扩展字段 ----
    speaker_name: Optional[str] = None   # 仅 AI听记 提供的说话人姓名推测
    is_overlap: bool = False             # 仅 AI听记 提供的重叠标记

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
```

**字段约束**：

| 字段 | 约束 | 违规处理 |
|:---|:---|:---|
| `start_ms` / `end_ms` | `0 <= start_ms < end_ms` | 抛出 `ValidationError` |
| `asr_confidence` | `[0.0, 1.0]` | 截断至边界值并记录 WARNING |
| `speaker_confidence` | `[0.0, 1.0]` | 截断至边界值并记录 WARNING |
| `text` | 非空，去除前后空白后 `len > 0` | 标记为空段，后续清洗阶段丢弃 |

#### 1.1.2 SessionRaw — 单次录音会话

```python
from datetime import datetime

@dataclass
class SessionRaw:
    """一次完整录音会话的原始数据容器。"""
    device_id: str                   # 设备标识，如 "dingtalk_a1_0032"
    device_type: str                 # 设备类型："dingtalk_a1" | "ai_tingji"
    session_start: datetime          # 会话起始时间（UTC+8）
    session_end: datetime            # 会话结束时间（UTC+8）
    segments: list[Segment]          # 按 start_ms 升序排列的片段列表
    source_file: Optional[str] = None  # 原始文件路径，用于溯源

    @property
    def total_duration_ms(self) -> int:
        return int((self.session_end - self.session_start).total_seconds() * 1000)

    @property
    def segment_count(self) -> int:
        return len(self.segments)
```

### 1.2 设备适配器接口

#### 1.2.1 基类定义

```python
from abc import ABC, abstractmethod
from pathlib import Path

class DeviceAdapter(ABC):
    """设备适配器基类。将异构设备输出转换为统一 SessionRaw。"""

    @abstractmethod
    def parse(self, raw_data: dict | str | Path) -> SessionRaw:
        """
        解析设备原始输出，返回标准化 SessionRaw。

        Args:
            raw_data: 设备原始数据（JSON dict、文件路径或原始字符串）。
        Returns:
            SessionRaw 实例。
        Raises:
            DeviceParseError: 数据格式不符合预期时抛出。
        """
        ...

    @abstractmethod
    def device_type(self) -> str:
        """返回设备类型标识字符串。"""
        ...
```

#### 1.2.2 DingTalk A1 适配器

```python
class DingTalkA1Adapter(DeviceAdapter):
    """
    DingTalk A1 输出格式：
    {
      "device_id": "...",
      "session_start": "2026-03-27T09:00:00+08:00",
      "session_end": "2026-03-27T17:30:00+08:00",
      "segments": [
        {
          "start_ms": 0, "end_ms": 3200,
          "speaker_id": "spk_001",
          "text": "...",
          "asr_confidence": 0.87,
          "speaker_confidence": 0.92
        }, ...
      ]
    }
    """

    def device_type(self) -> str:
        return "dingtalk_a1"

    def parse(self, raw_data: dict) -> SessionRaw:
        # 1. 校验顶层必填字段
        # 2. 逐条解析 segments，缺失 alignment_quality 时默认 MEDIUM
        # 3. 按 start_ms 排序
        # 4. 返回 SessionRaw
        ...
```

**DingTalk A1 格式要点**：

- `alignment_quality` 字段可能缺失，默认填充 `AlignmentQuality.MEDIUM`。
- `speaker_confidence` 可能缺失，默认填充 `0.5`。

#### 1.2.3 AI听记适配器

```python
class AITingjiAdapter(DeviceAdapter):
    """
    AI听记额外字段：speaker_name（姓名推测）、is_overlap（重叠标记）。
    格式与 DingTalk A1 类似，但 segments 内多出上述两个字段。
    """

    def device_type(self) -> str:
        return "ai_tingji"

    def parse(self, raw_data: dict) -> SessionRaw:
        # 1. 校验顶层必填字段
        # 2. 逐条解析 segments，保留 speaker_name、is_overlap
        # 3. 对 is_overlap=True 的片段标记 alignment_quality = LOW
        # 4. 按 start_ms 排序
        # 5. 返回 SessionRaw
        ...
```

**AI听记格式要点**：

- `speaker_name` 为推测值，仅用于后续 L2 模糊匹配的输入线索，不可直接作为最终说话人标识。
- `is_overlap=True` 的片段自动将 `alignment_quality` 降级为 `LOW`。

#### 1.2.4 适配器注册与分发

```python
# 适配器工厂，根据 device_type 字符串分发
_ADAPTER_REGISTRY: dict[str, type[DeviceAdapter]] = {
    "dingtalk_a1": DingTalkA1Adapter,
    "ai_tingji": AITingjiAdapter,
}

def get_adapter(device_type: str) -> DeviceAdapter:
    cls = _ADAPTER_REGISTRY.get(device_type)
    if cls is None:
        raise UnsupportedDeviceError(f"未知设备类型: {device_type}")
    return cls()
```

### 1.3 输入校验规则

校验在 `SessionRaw` 构建完成后、进入预处理层之前执行。

```python
@dataclass
class ValidationResult:
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

def validate_session(session: SessionRaw) -> ValidationResult:
    """执行三项核心校验，返回校验结果。"""
    ...
```

#### 1.3.1 时间线连续性检查

```
对于 segments[i] 与 segments[i+1]:
  gap = segments[i+1].start_ms - segments[i].end_ms
  若 gap < -500ms:   → ERROR: 严重时间倒退
  若 -500ms <= gap < 0: → WARNING: 微小重叠，自动修正（见 1.3.2）
  若 gap > 300_000ms (5min): → WARNING: 存在长静默，记录日志
```

**参数**：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `MAX_BACKWARD_TOLERANCE_MS` | 500 | 允许的最大时间倒退量（毫秒） |
| `LONG_SILENCE_THRESHOLD_MS` | 300,000 | 超过此值记录长静默警告 |

#### 1.3.2 段落重叠检测与修正

```
对于 segments[i] 与 segments[i+1]:
  overlap = segments[i].end_ms - segments[i+1].start_ms
  若 overlap > 0 且 overlap <= 500ms:
    → 自动修正: segments[i].end_ms = segments[i+1].start_ms（截断前段尾部）
    → 记录 WARNING
  若 overlap > 500ms:
    → 标记为 ERROR: 严重重叠，可能是数据异常
    → 不自动修正，交由人工处理或丢弃该 session
```

#### 1.3.3 总时长校验

```
declared_duration = session.total_duration_ms
actual_span = segments[-1].end_ms - segments[0].start_ms
deviation = abs(declared_duration - actual_span) / declared_duration

若 deviation > 0.1 (10%): → WARNING: 声明时长与实际跨度偏差较大
若 declared_duration > 14 * 3600 * 1000 (14小时): → WARNING: 超长录音
若 declared_duration < 60 * 1000 (1分钟): → WARNING: 过短录音
若 len(segments) == 0: → ERROR: 空会话
```

### 1.4 异常处理策略

| 异常类型 | 处理方式 | 后续动作 |
|:---|:---|:---|
| `DeviceParseError` | 适配器无法解析原始数据 | 记录错误日志，跳过该文件，通知运维 |
| `UnsupportedDeviceError` | 未知设备类型 | 拒绝处理，返回明确错误信息 |
| `ValidationError`（ERROR 级） | 数据存在不可自动修复的严重问题 | 跳过该 session，写入异常队列待人工处理 |
| `ValidationResult.warnings` | 数据存在可容忍的轻微问题 | 记录日志，继续处理 |

---

## 二、预处理层（core/preprocessor.py）

### 2.0 预处理流水线总览

```mermaid
flowchart TD
    A[SessionRaw 输入] --> B[Step 1: 对话会话检测]
    B --> C["List[ConversationSession]"]
    C --> D[Step 2: 置信度分级过滤]
    D --> E[Step 3: 话语合并]
    E --> F[Step 4: 特征计算]
    F --> G[Step 5: 关键人元数据关联]
    G --> H["List[ProcessedSession] 输出"]
```

预处理层的输入为校验通过的 `SessionRaw`，输出为富含特征标注的 `ProcessedSession` 列表。

```python
class Preprocessor:
    def __init__(self, config: PreprocessorConfig, key_people: KeyPeopleConfig):
        self.config = config
        self.key_people = key_people

    def process(self, session: SessionRaw) -> list[ProcessedSession]:
        # Step 1: 会话检测 — 切分为多个 ConversationSession
        conversations = self._detect_conversations(session)
        results = []
        for conv in conversations:
            # Step 2: 置信度过滤（需要关键人信息做分级）
            cleaned = self._filter_by_confidence(conv)
            # Step 3: 话语合并
            utterances = self._merge_utterances(cleaned)
            # Step 4: 特征计算
            enriched = self._compute_features(utterances, conv)
            # Step 5: 关键人元数据关联
            linked = self._link_key_people(enriched)
            results.append(ProcessedSession(
                session_id=conv.session_id,
                time_range=(conv.start_time, conv.end_time),
                utterances=linked,
            ))
        return results
```

### 2.1 Step 1：对话会话检测

#### 2.1.1 目标

将一段长时间连续录音（如 8-12 小时）按自然对话边界切分为独立的 `ConversationSession`，作为后续所有处理的基本单位。

#### 2.1.2 算法：双条件门控

```
输入: segments: List[Segment]（已按 start_ms 排序）
输出: conversations: List[ConversationSession]

当前会话 current = [segments[0]]
for i in range(1, len(segments)):
    gap_ms = segments[i].start_ms - segments[i-1].end_ms
    speaker_changed = segments[i].speaker_id != segments[i-1].speaker_id

    if gap_ms >= SILENCE_THRESHOLD_MS and speaker_changed:
        # 双条件同时满足 → 会话边界
        conversations.append(build_conversation(current))
        current = [segments[i]]
    else:
        current.append(segments[i])

conversations.append(build_conversation(current))  # 最后一个会话
```

**双条件门控的设计理由**：仅靠静默时长会将"一个人长时间思考后继续发言"误判为会话边界；仅靠说话人变化无法区分同一会议内的正常轮流发言。两个条件同时满足才切分。

#### 2.1.3 关键参数

| 参数 | 默认值 | 可配置 | 说明 |
|:---|:---|:---|:---|
| `SILENCE_THRESHOLD_MS` | 180,000 (3min) | 是 | 静默时长阈值 |
| `MIN_CONVERSATION_SEGMENTS` | 2 | 是 | 少于此数的会话合并到相邻会话 |

#### 2.1.4 输出数据结构

```python
@dataclass
class ConversationSession:
    session_id: str               # 自动生成，格式 "{device_id}_{序号:03d}"
    start_time: datetime          # 会话起始绝对时间
    end_time: datetime            # 会话结束绝对时间
    segments: list[Segment]       # 属于本会话的片段集合
    gap_before_ms: Optional[int]  # 与前一个会话的间隔（首个会话为 None）
```

#### 2.1.5 异常处理

| 场景 | 处理 |
|:---|:---|
| 全部片段构成单一会话（无边界） | 正常返回，仅包含一个 `ConversationSession` |
| 检测到极短会话（<2 个片段） | 合并至时间最近的相邻会话，记录 WARNING |

---

### 2.2 Step 2：置信度分级过滤

#### 2.2.1 目标

根据说话人的关键人等级，分级过滤低置信度片段。关键人的发言采用更宽松的阈值，避免丢失嘈杂环境下的关键指令。

#### 2.2.2 算法

```
输入: segments: List[Segment], key_people: KeyPeopleConfig
输出: filtered: List[Segment]

# 阈值映射表
CONFIDENCE_THRESHOLDS = {
    "P0": 0.35,
    "P1": 0.40,
    "P2": 0.50,
    "P3": 0.60,    # 非关键人同级
    "UNKNOWN": 0.60,
}

for seg in segments:
    level = lookup_speaker_level(seg.speaker_id, key_people)
    # level 为 "P0" | "P1" | "P2" | "P3" | "UNKNOWN"
    threshold = CONFIDENCE_THRESHOLDS[level]

    if seg.asr_confidence >= threshold:
        filtered.append(seg)
    else:
        # 记录被过滤的片段（含原因）到 filter_log
        filter_log.append(FilteredRecord(seg, reason="low_asr_confidence",
                                          threshold=threshold, level=level))
```

**注意**：此步骤需要提前执行一次粗粒度的说话人-关键人匹配（仅 L1 精确匹配），以获得说话人等级。完整的四级匹配在 Step 5 执行。若 L1 匹配不到，视为 `UNKNOWN`，使用最严格阈值 0.60。

#### 2.2.3 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `P0_ASR_THRESHOLD` | 0.35 | P0 关键人 ASR 置信度阈值 |
| `P1_ASR_THRESHOLD` | 0.40 | P1 关键人 ASR 置信度阈值 |
| `P2_ASR_THRESHOLD` | 0.50 | P2 关键人 ASR 置信度阈值 |
| `DEFAULT_ASR_THRESHOLD` | 0.60 | P3 及未知说话人 ASR 置信度阈值 |
| `COLD_START_THRESHOLD_OVERRIDE` | 0.25 | 冷启动期（Day 1-5）全局覆盖阈值 |

#### 2.2.4 异常处理

| 场景 | 处理 |
|:---|:---|
| 过滤后某会话片段为空 | 整个 `ConversationSession` 标记为 `LOW_QUALITY`，跳过后续步骤 |
| 过滤比例 >80% | 记录 WARNING，可能表明录音环境极差或设备故障 |

---

### 2.3 Step 3：话语合并

#### 2.3.1 目标

将同一说话人的连续短碎片聚合为语义连贯的 `Utterance`，减少碎片化，同时通过时长上限保证下游分片的灵活性。

#### 2.3.2 算法

```
输入: segments: List[Segment]（已过滤）
输出: utterances: List[Utterance]

current_group = [segments[0]]

for i in range(1, len(segments)):
    prev = segments[i - 1]
    curr = segments[i]
    gap_ms = curr.start_ms - prev.end_ms
    same_speaker = curr.speaker_id == prev.speaker_id
    group_duration_ms = curr.end_ms - current_group[0].start_ms

    can_merge = (
        same_speaker
        and gap_ms <= MERGE_GAP_THRESHOLD_MS
        and group_duration_ms <= MERGE_MAX_DURATION_MS
    )

    if can_merge:
        current_group.append(curr)
    else:
        utterances.append(build_utterance(current_group))
        current_group = [curr]

utterances.append(build_utterance(current_group))  # 最后一组
```

**`build_utterance` 合并逻辑**：

```python
def build_utterance(group: list[Segment]) -> Utterance:
    return Utterance(
        start_ms=group[0].start_ms,
        end_ms=group[-1].end_ms,
        speaker_id=group[0].speaker_id,
        text=" ".join(seg.text for seg in group),  # 合并文本，空格连接
        avg_asr_confidence=mean(seg.asr_confidence for seg in group),
        min_asr_confidence=min(seg.asr_confidence for seg in group),
        avg_speaker_confidence=mean(seg.speaker_confidence for seg in group),
        worst_alignment=min(seg.alignment_quality for seg in group, key=quality_rank),
        segment_count=len(group),
        has_overlap=any(seg.is_overlap for seg in group),
    )
```

#### 2.3.3 Utterance 数据结构

```python
@dataclass
class Utterance:
    start_ms: int
    end_ms: int
    speaker_id: str
    text: str
    # ---- 聚合置信度 ----
    avg_asr_confidence: float
    min_asr_confidence: float
    avg_speaker_confidence: float
    worst_alignment: AlignmentQuality
    # ---- 合并元信息 ----
    segment_count: int          # 合并前的原始 Segment 数量
    has_overlap: bool           # 是否包含重叠标记的 Segment
    # ---- 以下字段在 Step 4/5 中填充 ----
    features: Optional[UtteranceFeatures] = None
    key_person_match: Optional[KeyPersonMatch] = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
```

#### 2.3.4 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `MERGE_GAP_THRESHOLD_MS` | 4,000 (4s) | 同一说话人片段间隔 <=此值则合并（PRD 范围 3-5s，取中值） |
| `MERGE_MAX_DURATION_MS` | 120,000 (120s) | 单个 Utterance 时长上限，超过强制拆分 |

#### 2.3.5 异常处理

| 场景 | 处理 |
|:---|:---|
| 单个原始 Segment 已超过 120s | 按 120s 边界强制切分，切分点选取最近的句号/问号/感叹号位置 |
| 合并后文本为空（全部 Segment 文本为空白） | 丢弃该 Utterance，记录 WARNING |

---

### 2.4 Step 4：特征计算

#### 2.4.1 目标

为每个 `Utterance` 附加结构化特征，供下游智能分层和重要性评估使用。

#### 2.4.2 特征字段定义

```python
@dataclass
class UtteranceFeatures:
    # ---- 时间特征 ----
    duration_sec: float              # 时长（秒）
    start_time_of_day: str           # 绝对时间标记，如 "14:30"
    time_period: str                 # 时段标记："morning" | "afternoon" | "evening"
    position_in_session: float       # 在当前会话中的相对位置 [0.0, 1.0]

    # ---- 说话人特征 ----
    speaker_id: str                  # 说话人标识
    turn_index: int                  # 在当前会话中的轮次序号（从 1 开始）
    speaker_turn_count: int          # 该说话人在本会话中的累计发言次数
    speaker_duration_ratio: float    # 该说话人在本会话中的时长占比

    # ---- 文本统计特征 ----
    char_count: int                  # 文本字符数
    char_rate: float                 # 语速（字/秒）
    sentence_count: int              # 句子数（按句号/问号/感叹号切分）

    # ---- 内容特征 ----
    keyword_density: float           # 关键词命中密度（命中数 / 字符数）
    contains_question: bool          # 是否包含疑问句
    contains_action_word: bool       # 是否包含行动词（"决定""安排""要求""通知"等）
    scene_guess: Optional[str]       # 场景推测："meeting" | "call" | "dictation" | None
```

#### 2.4.3 计算逻辑伪代码

```python
def _compute_features(
    self,
    utterances: list[Utterance],
    conv: ConversationSession,
) -> list[Utterance]:

    # 预计算会话级统计
    total_duration = conv.end_time - conv.start_time
    speaker_durations: dict[str, int] = defaultdict(int)
    speaker_turn_counters: dict[str, int] = defaultdict(int)

    for utt in utterances:
        speaker_durations[utt.speaker_id] += utt.duration_ms

    for idx, utt in enumerate(utterances):
        speaker_turn_counters[utt.speaker_id] += 1

        utt.features = UtteranceFeatures(
            duration_sec=utt.duration_ms / 1000.0,
            start_time_of_day=format_time(conv.start_time, utt.start_ms),
            time_period=classify_time_period(conv.start_time, utt.start_ms),
            position_in_session=utt.start_ms / max(1, conv_duration_ms),
            speaker_id=utt.speaker_id,
            turn_index=idx + 1,
            speaker_turn_count=speaker_turn_counters[utt.speaker_id],
            speaker_duration_ratio=(
                speaker_durations[utt.speaker_id] / max(1, sum(speaker_durations.values()))
            ),
            char_count=len(utt.text),
            char_rate=len(utt.text) / max(0.1, utt.duration_ms / 1000.0),
            sentence_count=count_sentences(utt.text),
            keyword_density=compute_keyword_density(utt.text, KEYWORD_LIST),
            contains_question="？" in utt.text or "吗" in utt.text,
            contains_action_word=any(w in utt.text for w in ACTION_WORDS),
            scene_guess=guess_scene(utt.text),
        )
    return utterances
```

#### 2.4.4 关键参数与辅助常量

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `KEYWORD_LIST` | 配置文件加载 | 业务关键词列表（"预算""决策""风险""合同"等） |
| `ACTION_WORDS` | `["决定", "安排", "要求", "通知", "确认", "同意", "否决", "推迟"]` | 行动词集合 |
| `TIME_PERIOD_BOUNDARIES` | `{"morning": (6,12), "afternoon": (12,18), "evening": (18,24)}` | 时段划分边界（小时） |

#### 2.4.5 `classify_time_period` 逻辑

```
hour = (session_start + offset_ms).hour
if 6 <= hour < 12:  return "morning"
if 12 <= hour < 18: return "afternoon"
return "evening"
```

---

### 2.5 Step 5：关键人元数据关联

#### 2.5.1 目标

将每个 Utterance 的 `speaker_id` 关联到关键人配置，输出匹配结果与匹配层级。

#### 2.5.2 处理顺序：L1 → L1.5 → L2 → L3 逐层递进

```mermaid
flowchart TD
    S[Utterance.speaker_id] --> L1{L1: Speaker ID 精确匹配}
    L1 -- 命中 --> R1[match_type=exact, level=配置等级]
    L1 -- 未命中 --> L15{L1.5: ASR 人名纠错匹配}
    L15 -- 命中 --> R15[match_type=asr_corrected, level=配置等级]
    L15 -- 未命中 --> L2{L2: 姓名/别名模糊匹配}
    L2 -- 命中 --> R2[match_type=alias, level=配置等级]
    L2 -- 未命中 --> L3{L3: 内容推断匹配}
    L3 -- 命中 --> R3["match_type=inferred, level=降一级"]
    L3 -- 未命中 --> RN[match_type=none, level=UNKNOWN]
```

#### 2.5.3 各层匹配逻辑

**L1 — Speaker ID 精确匹配**：
```
for person in key_people:
    if utterance.speaker_id in person.speaker_ids:
        return KeyPersonMatch(person_id=person.id, level=person.level,
                              match_type="exact", confidence=0.98)
```

**L1.5 — ASR 人名纠错匹配**：
```
extracted_names = extract_names_from_text(utterance.text)
for name in extracted_names:
    for correction in asr_corrections:
        if name in correction.variants:
            person = find_person_by_name(correction.target)
            if person:
                return KeyPersonMatch(person_id=person.id, level=person.level,
                                      match_type="asr_corrected", confidence=0.92)
```

**L2 — 姓名/别名模糊匹配**：
```
# 利用 speaker_name（AI听记提供）或文本中的自报姓名
candidate_name = utterance.speaker_name or extract_self_introduction(utterance.text)
if candidate_name:
    for person in key_people:
        if fuzzy_match(candidate_name, person.name, person.aliases, threshold=0.8):
            return KeyPersonMatch(person_id=person.id, level=person.level,
                                  match_type="alias", confidence=0.87)
```

**L3 — 内容推断匹配**：
```
# 基于上下文语义推断：如"总裁指示..."中虽无直呼姓名但可推断
inferred_person = infer_speaker_from_context(utterance.text, key_people)
if inferred_person:
    effective_level = downgrade_level(inferred_person.level)  # P0→P1, P1→P2, P2→P3
    return KeyPersonMatch(person_id=inferred_person.id, level=effective_level,
                          match_type="inferred", confidence=0.75)
```

#### 2.5.4 匹配结果数据结构

```python
@dataclass
class KeyPersonMatch:
    person_id: Optional[str]     # 关键人 ID，未匹配时为 None
    person_name: Optional[str]   # 关键人姓名
    level: str                   # 有效等级："P0" | "P1" | "P2" | "P3" | "UNKNOWN"
    original_level: Optional[str]  # 原始等级（仅 inferred 时与 level 不同）
    match_type: str              # "exact" | "asr_corrected" | "alias" | "inferred" | "none"
    confidence: float            # 匹配置信度
```

#### 2.5.5 异常处理

| 场景 | 处理 |
|:---|:---|
| 同一 speaker_id 在不同 Utterance 匹配到不同关键人 | 取置信度最高的匹配结果全局统一，记录 WARNING |
| L3 推断匹配到多个候选人 | 取置信度最高者，若差距 <0.1 则标记 `ambiguous`，不做匹配 |
| 关键人配置为空 | 所有 Utterance 的 `key_person_match.match_type` 均为 `none`，正常继续 |

---

### 2.6 预处理层最终输出

```python
@dataclass
class ProcessedSession:
    session_id: str                          # 会话标识
    time_range: tuple[datetime, datetime]    # 起止时间
    utterances: list[Utterance]              # 已关联特征与关键人信息的 Utterance 列表
    metadata: SessionMetadata                # 会话级聚合元信息

@dataclass
class SessionMetadata:
    total_duration_sec: float
    utterance_count: int
    speaker_ids: list[str]
    key_person_ids: list[str]           # 本会话中匹配到的关键人 ID 列表
    dominant_speaker_id: Optional[str]  # 发言时长最长的说话人
    scene_guess: Optional[str]          # 会话级场景推测（取 Utterance 中出现频率最高的场景）
    filtered_segment_count: int         # Step 2 中被过滤掉的片段数
    avg_asr_confidence: float           # 会话整体平均 ASR 置信度
```

---

### 2.7 配置汇总（PreprocessorConfig）

```python
@dataclass
class PreprocessorConfig:
    # Step 1: 会话检测
    silence_threshold_ms: int = 180_000       # 3 分钟
    min_conversation_segments: int = 2

    # Step 2: 置信度过滤
    p0_asr_threshold: float = 0.35
    p1_asr_threshold: float = 0.40
    p2_asr_threshold: float = 0.50
    default_asr_threshold: float = 0.60
    cold_start_threshold: float = 0.25
    is_cold_start: bool = False               # 冷启动模式开关

    # Step 3: 话语合并
    merge_gap_threshold_ms: int = 4_000       # 4 秒
    merge_max_duration_ms: int = 120_000      # 120 秒

    # Step 4: 特征计算
    keyword_list_path: str = "config/keywords.yaml"
    time_period_boundaries: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "morning": (6, 12),
            "afternoon": (12, 18),
            "evening": (18, 24),
        }
    )

    # Step 5: 关键人匹配
    asr_corrections_path: str = "config/asr_name_corrections.yaml"
    fuzzy_match_threshold: float = 0.80       # L2 模糊匹配相似度阈值
    inferred_match_min_confidence: float = 0.70  # L3 推断匹配最低置信度
```

---

### 2.8 全局异常处理策略

| 层级 | 策略 | 说明 |
|:---|:---|:---|
| 单个 Segment 异常 | 跳过该 Segment，记录日志 | 不影响其余数据处理 |
| 单个 ConversationSession 全部被过滤 | 标记 `LOW_QUALITY`，保留元信息但不输出 Utterance | 日报中备注"存在低质量录音段" |
| 预处理层整体异常（如配置加载失败） | 阻止流水线启动，返回明确错误 | 不产出任何结果，避免静默失败 |
| 关键人配置加载失败 | 使用上一次成功加载的配置继续处理 | 记录 ERROR 日志，通知运维 |
