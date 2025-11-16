# ShareGPT 格式、工具定义与 Loss Mask 完整指南

## 1. 工具定义转化为 ShareGPT 格式

### 1.1 工具定义的基本结构

工具定义通常采用 JSON Schema 格式，包含以下字段：

```json
{
  "name": "function_name",
  "description": "函数描述",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {"type": "string", "description": "参数1描述"},
      "param2": {"type": "number", "description": "参数2描述"}
    },
    "required": ["param1"]
  }
}
```

### 1.2 将 ShareGPT 格式转化的过程

在 LLaMA-Factory 中，工具定义通过以下步骤将的 ShareGPT 格式转化成为训练格式：

原始数据文件 (JSON/JSONL/CSV)
    ↓
DatasetConverter 自动转化
    ↓
统一内部格式：
  {
    "_prompt": [...],
    "_response": [...],
    "_system": "...",
    "_tools": "...",
    "_images": [...],
    "_videos": [...],
    "_audios": [...]
  }
    ↓
模板格式化 (Template)
    ↓
训练

**步骤 1：数据转换**
- 位置：`src/llamafactory/data/converter.py` 中的 `SharegptDatasetConverter` 类
- 工具定义存储在 `_tools` 字段中（JSON 字符串格式）

**步骤 2：模板格式化**
- 位置：`src/llamafactory/data/formatter.py` 中的 `ToolFormatter` 类
- 调用 `tool_utils.tool_formatter()` 将工具列表转化为模型特定的格式

**步骤 3：模型特定的工具格式**
- 位置：`src/llamafactory/data/tool_utils.py`
- 支持多种模型的工具格式：
  - **Qwen2.5**: `QwenToolUtils` - JSON 格式的工具定义
  - **GLM-4**: `GLM4ToolUtils` - 特定的 GLM 格式
  - **Llama3**: `Llama3ToolUtils` - Llama 特定格式
  - **Mistral**: `MistralToolUtils` - Mistral 特定格式

### 1.3 ShareGPT 数据格式示例

```json
[
  {
    "conversations": [
      {"from": "human", "value": "用户指令"},
      {"from": "function_call", "value": "{\"name\": \"func_name\", \"arguments\": {...}}"},
      {"from": "observation", "value": "工具返回结果"},
      {"from": "gpt", "value": "模型回答"}
    ],
    "system": "系统提示词（可选）",
    "tools": "[{\"name\": \"func_name\", \"description\": \"...\", \"parameters\": {...}}]"
  }
]
```

**关键点**：
- `conversations` 中 human/observation 必须在奇数位置，gpt/function_call 在偶数位置
- `tools` 字段包含 JSON 字符串格式的工具定义列表
- 工具调用结果通过 `observation` 角色返回

---

## 2. ShareGPT 格式的 Loss Mask 实现

### 2.1 Loss Mask 的基本概念

Loss Mask 用于控制哪些 token 参与损失计算：
- **IGNORE_INDEX (-100)**：该位置的 token 不计算损失
- **其他值**：该位置的 token 计算损失

### 2.2 Loss Mask 的实现位置

**主要文件**：`src/llamafactory/data/processor/supervised.py`

```python
# 关键代码片段
if self.data_args.train_on_prompt:
    source_label = source_ids  # 提示词参与损失计算
else:
    source_label = [IGNORE_INDEX] * source_len  # 提示词不计算损失

if self.data_args.mask_history and turn_idx != 0:
    target_label = [IGNORE_INDEX] * target_len  # 历史回答不计算损失
else:
    target_label = target_ids  # 当前回答计算损失
```

### 2.3 Loss Mask 的配置参数

在训练配置中可以控制：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `train_on_prompt` | 是否在提示词上计算损失 | False |
| `mask_history` | 是否 mask 历史对话 | False |
| `ignore_pad_token_for_loss` | 是否忽略 padding token | True |

### 2.4 Loss Mask 的计算流程

1. **编码阶段**（`_encode_data_example`）：
   - 将 prompt 和 response 分别编码
   - 根据配置生成对应的 label 序列

2. **标签生成**：
   ```
   input_ids:  [prompt_tokens] + [response_tokens]
   labels:     [IGNORE_INDEX...] + [response_tokens]  # 默认配置
   ```

3. **数据整理**（`SFTDataCollatorWith4DAttentionMask`）：
   - 位置：`src/llamafactory/data/collator.py`
   - 将 labels 中的 IGNORE_INDEX 替换为 pad_token_id
   - 在损失计算时自动忽略这些位置

---

## 3. reasoning_content 支持情况

### 3.1 当前支持状态

**支持情况**：部分支持，通过 `enable_thinking` 参数实现

### 3.2 实现机制

**位置**：`src/llamafactory/data/template.py` 中的 `ReasoningTemplate` 类

```python
# 思维链的处理
if enable_thinking is True:  # 慢思考
    # 空思维链添加到回答中，计算损失
    response_ids = self.get_thought_word_ids(tokenizer) + response_ids
elif enable_thinking is False:  # 快思考
    # 空思维链添加到提示词中，不计算损失
    prompt_ids += self.get_thought_word_ids(tokenizer)
```

### 3.3 配置方式

在训练配置中设置：

```yaml
template: "qwen3"  # 或其他支持推理的模型
enable_thinking: true  # true=慢思考, false=快思考, null=混合
```

### 3.4 限制说明

- **不支持独立的 reasoning_content 字段**
- 思维链必须包含在 response 的内容中
- 格式：`<think>思维过程</think>最终回答`
- 仅支持特定模型（Qwen3、GLM-4V-MOE 等）

---

## 4. 最佳实践建议

### 4.1 数据准备

1. 确保工具定义为有效的 JSON 格式
2. 验证 conversations 中角色的位置正确性
3. 工具调用和观察必须成对出现

### 4.2 Loss Mask 配置

- 通常保持默认配置（不在提示词上计算损失）
- 对于多轮对话，建议启用 `mask_history`
- 根据任务需求调整 `train_on_prompt`

### 4.3 模型选择

- Qwen2.5-7B-Instruct：使用 `qwen` 工具格式
- 其他模型：查看 `tool_utils.py` 中的 TOOLS 字典

---

## 5. 代码实现细节

### 5.1 工具格式化流程

**Qwen 工具格式化示例**（`tool_utils.py` 第 290-301 行）：

```python
class QwenToolUtils(ToolUtils):
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" \
                else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)
        return QWEN_TOOL_PROMPT.format(tool_text=tool_text)
```

### 5.2 Loss 计算中的 Mask 应用

**数据整理器**（`collator.py` 第 155-164 行）：

```python
if self.tokenizer.padding_side == "right":
    features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
    features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
    features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
```

### 5.3 多轮对话的 Loss Mask

**处理逻辑**（`supervised.py` 第 63-80 行）：

```python
if self.data_args.train_on_prompt:
    source_label = source_ids
elif self.template.efficient_eos and turn_idx != 0:
    source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
else:
    source_label = [IGNORE_INDEX] * source_len

if self.data_args.mask_history and turn_idx != 0:
    target_label = [IGNORE_INDEX] * target_len
else:
    target_label = target_ids
```

---

## 6. 常见问题解答

### Q1: 如何验证工具定义是否正确转化？

**A**: 检查以下几点：
1. 在 `dataset_info.json` 中配置 `formatting: "sharegpt"`
2. 确保 `tools` 字段包含有效的 JSON 字符串
3. 运行数据预处理，检查输出的 `_tools` 字段

### Q2: Loss Mask 为什么使用 -100？

**A**: PyTorch 的 CrossEntropyLoss 默认忽略 index 为 -100 的位置。这是标准约定，在 `extras/constants.py` 中定义为 `IGNORE_INDEX = -100`。

### Q3: 如何在 Qwen2.5 上使用工具调用？

**A**:
1. 设置 `template: "qwen"`
2. 在 ShareGPT 数据中包含 `function_call` 角色
3. 工具定义使用 JSON Schema 格式
4. 模型会自动格式化工具调用

### Q4: reasoning_content 和思维链的区别？

**A**:
- **reasoning_content**：OpenAI 的标准字段（当前不支持）
- **思维链**：通过 `<think>...</think>` 标签实现，支持 enable_thinking 参数

### Q5: 如何在多轮对话中只训练最后一轮？

**A**: 设置 `mask_history: true`，这样只有最后一轮的回答会计算损失。

---

## 7. 参考文件位置

| 功能 | 文件位置 |
|------|--------|
| 数据转换 | `src/llamafactory/data/converter.py` |
| 模板定义 | `src/llamafactory/data/template.py` |
| 工具处理 | `src/llamafactory/data/tool_utils.py` |
| 格式化器 | `src/llamafactory/data/formatter.py` |
| 数据处理 | `src/llamafactory/data/processor/supervised.py` |
| 数据整理 | `src/llamafactory/data/collator.py` |
| 常量定义 | `src/llamafactory/extras/constants.py` |
| 示例数据 | `data/glaive_toolcall_zh_demo.json` |

---

## 8. 总结

### ShareGPT 格式优势
- ✅ 支持多角色对话
- ✅ 灵活的工具调用支持
- ✅ 完整的观察-反思循环
- ✅ 适合复杂的多轮交互

### Loss Mask 关键点
- ✅ 通过 IGNORE_INDEX (-100) 实现
- ✅ 在数据处理阶段生成
- ✅ 在数据整理阶段应用
- ✅ 支持灵活的掩码策略

### 工具支持现状
- ✅ 多种模型的工具格式
- ✅ 自动工具定义转化
- ✅ 部分推理能力支持
- ⚠️ reasoning_content 需要自定义实现


