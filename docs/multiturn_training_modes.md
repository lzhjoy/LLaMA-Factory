# LLaMA-Factory 多轮对话训练的两种模式

## 核心问题

LLaMA-Factory 对于多轮对话的训练有两种模式：
1. **模式 1**：`mask_history=False`（默认）- 每轮独立训练
2. **模式 2**：`mask_history=True` - 仅最后一轮训练

## 关键发现：encode_multiturn 生成的是相邻配对，而不是累积配对！

### encode_multiturn 的实现

```python
# src/llamafactory/data/template.py 第 83 行
def encode_multiturn(self, tokenizer, messages, system, tools):
    encoded_messages = self._encode(tokenizer, messages, system, tools)
    # 关键：相邻配对，不是累积配对！
    return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]
```

### 示例：3 轮对话

**输入消息**：
```
messages = [
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2"},
    {"role": "user", "content": "Q3"},
    {"role": "assistant", "content": "A3"},
]
```

**_encode 输出**（每个消息单独编码）：
```
encoded_messages = [
    Q1_tokens,      # index 0
    A1_tokens,      # index 1
    Q2_tokens,      # index 2
    A2_tokens,      # index 3
    Q3_tokens,      # index 4
    A3_tokens,      # index 5
]
```

**encode_multiturn 输出**（相邻配对）：
```
encoded_pairs = [
    (Q1_tokens, A1_tokens),    # Pair 1: i=0
    (Q2_tokens, A2_tokens),    # Pair 2: i=2
    (Q3_tokens, A3_tokens),    # Pair 3: i=4
]
```

---

## 模式 1：mask_history=False（默认）

### 处理流程

在 `supervised.py` 第 79-80 行，顺序拼接所有 pair：

```python
for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
    source_label = [IGNORE_INDEX] * len(source_ids)  # Mask prompt
    target_label = target_ids                         # 计算 response 损失
    
    input_ids += source_ids + target_ids
    labels += source_label + target_label
```

### 最终结果

```
input_ids:  [Q1] [A1] [Q2] [A2] [Q3] [A3]
labels:     [-100] [A1] [-100] [A2] [-100] [A3]
```

**特点**：
- ✅ 每轮都有训练信号
- ✅ 模型学习每一轮的独立回答能力
- ✅ 显存占用少

---

## 模式 2：mask_history=True

### 处理流程

在 `supervised.py` 第 49-77 行：

```python
# 步骤 1：反向处理 pair（高优先级给最后一轮）
encoded_pairs = encoded_pairs[::-1]  # [Pair3, Pair2, Pair1]

# 步骤 2：反向拼接
for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
    if turn_idx != 0:  # 非最后一轮
        target_label = [IGNORE_INDEX] * len(target_ids)  # Mask response
    else:  # 最后一轮
        target_label = target_ids  # 计算 response 损失
    
    # 反向拼接（prepend）
    input_ids = source_ids + target_ids + input_ids
    labels = source_label + target_label + labels
```

### 详细过程

**初始化**：`input_ids = []`, `labels = []`

**第 1 次迭代**（Pair 3 - 最后一轮）：
```
input_ids = [Q3] [A3] + [] = [Q3] [A3]
labels = [-100] [A3] + [] = [-100] [A3]
```

**第 2 次迭代**（Pair 2 - 反向拼接）：
```
input_ids = [Q2] [A2] + [Q3] [A3] = [Q2] [A2] [Q3] [A3]
labels = [-100] [-100] + [-100] [A3] = [-100] [-100] [-100] [A3]
```

**第 3 次迭代**（Pair 1 - 反向拼接）：
```
input_ids = [Q1] [A1] + [Q2] [A2] [Q3] [A3] = [Q1] [A1] [Q2] [A2] [Q3] [A3]
labels = [-100] [-100] + [-100] [-100] [-100] [A3] = [-100] [-100] [-100] [-100] [-100] [A3]
```

### 最终结果

```
input_ids:  [Q1] [A1] [Q2] [A2] [Q3] [A3]
labels:     [-100] [-100] [-100] [-100] [-100] [A3]
```

**特点**：
- ✅ 只训练最后一轮
- ✅ 显存占用少
- ✅ 在 cutoff_len 限制下优先保留最后一轮

---

## 两种模式对比

| 方面 | mask_history=False | mask_history=True |
|------|-------------------|------------------|
| **Pair 生成** | 相邻配对 | 相邻配对（反向处理） |
| **拼接方式** | 顺序拼接 | 反向拼接 |
| **训练轮次** | 所有轮 | 仅最后一轮 |
| **input_ids** | [Q1][A1][Q2][A2][Q3][A3] | [Q1][-100][Q2][-100][Q3][A3] |
| **labels** | [-100][A1][-100][A2][-100][A3] | [-100][-100][-100][-100][-100][A3] |
| **显存占用** | 少 | 更少 |
| **训练信号** | 多 | 少 |

---

## 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| encode_multiturn | `src/llamafactory/data/template.py` | 74-83 |
| _encode | `src/llamafactory/data/template.py` | 129-166 |
| 模式 1 处理 | `src/llamafactory/data/processor/supervised.py` | 79-80 |
| 模式 2 处理 | `src/llamafactory/data/processor/supervised.py` | 49-77 |

---

## 常见问题

### Q1：为什么 encode_multiturn 生成相邻配对而不是累积配对？

**A**：因为 `_encode` 对每个消息单独编码，包括系统提示和工具定义。如果生成累积配对，会导致系统提示和工具定义被重复编码多次，浪费显存。相邻配对的设计更高效。

### Q2：mask_history=True 时为什么要反向拼接？

**A**：为了在 `cutoff_len` 限制下优先保留最后一轮。如果顺序拼接，可能会因为长度限制而丢弃最后一轮。反向拼接确保最后一轮总是被保留。

### Q3：为什么 Pair 2 的 source 是 [Q2] 而不是 [Q1,A1,Q2]？

**A**：因为 `_encode` 对每个消息单独编码。Pair 2 的 source 是 `encoded_messages[2]`，即第 3 个消息（Q2），不包含历史。

### Q4：那对话历史是如何被模型学到的？

**A**：通过 ShareGPT 格式的 `conversations` 字段。在数据预处理时，完整的对话历史被保留在 prompt 中，模型通过学习 prompt 来理解对话历史。

