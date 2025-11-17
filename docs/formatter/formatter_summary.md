# 数据格式转换器修改总结

## 修改内容

### 文件：`src/format/formater.py`

**原功能**：将 OpenAI 格式转换为 ShareGPT 格式

**新功能**：将任意格式规范化为标准 OpenAI 格式，可直接用于 LLaMA-Factory 训练

### 核心改动

#### 1. 删除 `_openai2sharegpt` 函数
- 不再转换为 ShareGPT 格式
- 改为规范化 OpenAI 格式

#### 2. 新增 `_normalize_openai_format` 函数

**功能**：
- 规范化 OpenAI 格式数据
- 处理 `tool_calls` 和 `function_call` 两种格式
- 确保 `arguments` 为 JSON 字符串
- 智能处理 `content` 字段（有 tool_calls 时设为 null）
- 保留 `tool_call_id` 字段

**输出格式**：
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [...]
    },
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

#### 3. 改进错误处理
- 详细的日志记录
- 跳过格式错误的消息
- 继续处理其他数据

#### 4. 支持多种输入格式
- JSONL 文件
- Parquet 文件
- 自动检测文件类型

## 使用方法

### 基础用法

```python
from src.format.formater import Formatter

formatter = Formatter(
    input_dir="data/input",
    output_path="data/output.jsonl"
)
formatter.format()
```

### 命令行用法

```bash
python examples/convert_to_openai_format.py \
    --input_dir data/raw \
    --output_path data/formatted.jsonl
```

## 支持的数据格式

### 输入格式

支持任何包含 `messages` 字段的格式：

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]}
  ]
}
```

### 输出格式

标准 OpenAI 格式（JSONL）：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## 在 LLaMA-Factory 中使用

### 1. 配置 dataset_info.json

```json
{
  "my_dataset": {
    "file_name": "data/formatted.jsonl",
    "formatting": "openai"
  }
}
```

### 2. 训练配置

```bash
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset my_dataset \
  --template qwen \
  --train_on_prompt false \
  --output_dir output
```

## 特性

✓ 自动规范化 tool_calls 格式  
✓ 支持 tool_calls 和 function_call 两种格式  
✓ 智能处理 content 字段  
✓ 保留 tool_call_id  
✓ 详细的日志记录  
✓ 错误恢复能力强  
✓ 支持 JSONL 和 Parquet 输入  

## 测试

所有功能已通过测试：
- ✓ tool_calls 格式规范化
- ✓ tool 消息处理
- ✓ 简单对话处理
- ✓ 多工具调用处理
- ✓ 错误处理和日志记录

