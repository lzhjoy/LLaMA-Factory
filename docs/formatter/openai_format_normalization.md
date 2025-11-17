# OpenAI 格式标准化

## 概述

`src/format/formater.py` 中的 `_normalize_openai_format` 函数用于标准化不规范的 OpenAI 格式数据。

## 支持的标准化操作

### 1. Role 标准化

- `"function"` → `"tool"`
- 保留标准角色：`"user"`, `"assistant"`, `"system"`, `"tool"`
- 大小写不敏感

### 2. Content 处理

- `None` → `""`（空字符串）
- 非字符串类型 → 转换为字符串
- 保留原始内容

### 3. tool_calls 标准化

支持多种格式的 `tool_calls`：

#### 格式 1：标准 OpenAI 格式
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"NY\"}"
      }
    }
  ]
}
```

#### 格式 2：tool_calls 为字符串
```json
{
  "role": "assistant",
  "tool_calls": "[{\"id\": \"call_123\", ...}]"
}
```

#### 格式 3：旧的 function_call 格式（自动转换）
```json
{
  "role": "assistant",
  "function_call": {
    "name": "calculator",
    "arguments": "{\"a\": 2, \"b\": 2}"
  }
}
```

转换为：
```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "arguments": {"a": 2, "b": 2}
      }
    }
  ]
}
```

### 4. Arguments 解析

自动解析 `arguments` 字段：
- JSON 字符串 → JSON 对象
- 多个 JSON 对象 → 列表
- 无法解析 → 保留原字符串

### 5. tool_call_id 处理

保留 `tool_call_id` 字段（用于 `role: "tool"` 的消息）

### 6. Messages 字符串解析

如果 `messages` 是 JSON 字符串，自动解析为列表

## 使用示例

```python
from src.format.formater import Formatter

formatter = Formatter(
    input_dir="data/input",
    output_path="data/output.jsonl"
)
formatter.format()
```

## 输入格式

支持：
- `.jsonl` 文件（每行一个 JSON 对象）
- `.parquet` 文件

## 输出格式

标准化后的 OpenAI 格式：
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## 错误处理

- 无效的 JSON 格式 → 记录警告，跳过该项
- 无效的 role → 记录警告，跳过该消息
- 无效的 tool_calls → 记录警告，忽略该字段
- 空消息列表 → 跳过该项

## 日志输出

- 成功处理的项数
- 各种解析错误和警告

