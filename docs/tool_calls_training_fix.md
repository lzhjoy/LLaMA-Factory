# Tool Calls 训练修复

## 问题

在之前的实现中，OpenAI 格式的 `tool_calls` 字段被转换为 JSON 字符串并放在 `_prompt` 中，这导致：

1. **tool_calls 被 mask 掉**：由于 `train_on_prompt=False`（默认值），prompt 部分的所有 token 都被设为 `-100`，包括 tool_calls
2. **模型无法学习调用工具**：模型只能学习如何根据工具结果生成回答，而不能学习如何调用工具

## 解决方案

### 1. 保留 `tool_calls` 字段（`src/llamafactory/data/converter.py`）

修改 `OpenAIDatasetConverter`，不再将 `tool_calls` 转换为 JSON 字符串，而是保留原始的 `tool_calls` 字段：

```python
# 之前：tool_calls 被转换为 JSON 字符串
if "tool_calls" in message and len(message["tool_calls"]) > 0:
    tool_calls_list = [tool["function"] for tool in message["tool_calls"]]
    content = json.dumps(tool_calls_list, ensure_ascii=False)
    role = self.dataset_attr.function_tag

# 现在：保留 tool_calls 字段
msg_dict = {
    "role": tag_mapping[role],
    "content": content,
}
if "tool_calls" in message:
    msg_dict["tool_calls"] = message["tool_calls"]
```

### 2. 使用 `apply_chat_template` 处理 tool_calls（`src/llamafactory/data/template.py`）

修改 `_encode` 方法，当消息包含 `tool_calls` 时，使用 `tokenizer.apply_chat_template` 而不是预定义的格式化器：

```python
# 检查是否有 tool_calls
has_tool_calls = any("tool_calls" in msg for msg in messages)

if has_tool_calls:
    # 使用 apply_chat_template 处理 tool_calls
    # 这样 tokenizer 的 chat_template 可以原生处理 tool_calls
    for i, message in enumerate(messages):
        msg_list = [...]  # 构建消息列表
        encoded_ids = tokenizer.apply_chat_template(msg_list, tokenize=True, add_generation_prompt=False)
        encoded_messages.append(encoded_ids)
```

### 3. 检测 tokenizer 对 tool_calls 的支持（`src/llamafactory/data/template.py`）

在 `parse_template` 中检测 tokenizer 是否原生支持 `tool_calls`，如果支持则使用相应的 `assistant_slot`：

```python
messages_with_tools = [
    {"role": "user", "content": "{{content}}"},
    {"role": "assistant", "content": "{{content}}", "tool_calls": [...]}
]
try:
    assistant_slot_with_tools = tokenizer.apply_chat_template(messages_with_tools, ...)
    if "<tool_call>" in assistant_slot_with_tools:
        assistant_slot = assistant_slot_with_tools  # 使用支持 tool_calls 的 slot
except Exception:
    pass  # 使用默认 slot
```

## 效果

### 关键改变

**之前**：
- `tool_calls` 被转换为 JSON 字符串，放在 `_prompt` 中
- 由于 `train_on_prompt=False`，tool_calls 被 mask 掉，模型无法学习

**现在**：
- `tool_calls` 保留为原始字段，由 `tokenizer.apply_chat_template` 原生处理
- `tool_calls` 被放在 response 中，模型可以学习

### 消息结构（不变）

```json
{
  "role": "assistant",
  "content": null,  // 或者有内容
  "tool_calls": [...]  // 保留原始字段
}
```

tokenizer 的 `apply_chat_template` 会原生处理这个结构，生成格式化的输出。

### Loss Mask 行为

对于包含 tool_calls 的对话：

```
Pair 1: user_question → tool_calls (trained!)
Pair 2: observation → final_answer (trained!)
```

- **Pair 1**：
  - Prompt: user_question (masked)
  - Response: tool_calls (trained) ✓

- **Pair 2**：
  - Prompt: observation (masked)
  - Response: final_answer (trained) ✓

## 使用方式

无需改变任何配置，只需确保：

1. 数据格式包含 `tool_calls` 字段
2. 使用支持 `tool_calls` 的 tokenizer（如 Qwen2.5）
3. 设置 `train_on_prompt=False`（默认值）

## 验证

运行以下命令验证 tool_calls 被正确训练：

```bash
# 查看 tool_calls 是否出现在 response 中
python -c "
from transformers import AutoTokenizer
from src.llamafactory.data.template import get_template_and_fix_tokenizer
from src.llamafactory.hparams import DataArguments

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template='qwen'))

messages = [
    {'role': 'user', 'content': 'What\\'s the weather?'},
    {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {'name': 'get_weather', 'arguments': '{}'}}]},
]

pairs = template.encode_multiturn(tokenizer, messages)
for i, (prompt, response) in enumerate(pairs):
    print(f'Pair {i}: tool_calls in response = {\"tool_call\" in tokenizer.decode(response).lower()}')
"
```

