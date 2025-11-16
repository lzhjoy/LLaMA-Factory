# Qwen2.5-7B-Instruct Agent 数据全量微调完整计划

## 概述

本计划详细说明如何对 Qwen2.5-7B-Instruct 进行包含 Agent 数据（工具调用 + reasoning_content）的全量微调。整个流程分为三个主要阶段：

1. **数据转化** - 将原始 Agent 数据转化为 LLaMA-Factory 可用的 ShareGPT 格式
2. **配置训练参数** - 编写训练配置文件
3. **执行训练** - 运行微调任务

---

## 第一阶段：数据转化为 ShareGPT 格式

### 1.1 数据格式要求

Agent 数据需要包含以下内容：
- **工具定义**：JSON Schema 格式的工具描述
- **工具调用**：模型生成的函数调用
- **工具观察**：函数执行结果
- **推理内容**：思维链（可选，使用 `<think>...</think>` 标签）

### 1.2 数据转化步骤

**步骤 1：准备原始数据**

创建 `data/agent_data.json`，采用 ShareGPT 格式：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "用户指令"},
      {"from": "function_call", "value": "{\"name\": \"func\", \"arguments\": {...}}"},
      {"from": "observation", "value": "工具返回结果"},
      {"from": "gpt", "value": "<think>思维过程</think>最终回答"}
    ],
    "system": "系统提示词（可选）",
    "tools": "[{\"name\": \"func\", \"description\": \"...\", \"parameters\": {...}}]"
  }
]
```

**关键点**：
- `conversations` 中 human/observation 必须在奇数位置
- `function_call` 和 `gpt` 必须在偶数位置
- `tools` 字段为 JSON 字符串格式
- 推理内容包含在 `gpt` 的 value 中，使用 `<think>...</think>` 标签

**步骤 2：注册数据集**

编辑 `data/dataset_info.json`，添加：

```json
"agent_data": {
  "file_name": "agent_data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}
```

**步骤 3：验证数据**

运行数据预处理检查：

```bash
python src/llamafactory/cli.py \
  --stage sft \
  --do_train false \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset agent_data \
  --template qwen \
  --cutoff_len 2048 \
  --overwrite_cache
```

---

## 第二阶段：配置训练参数文件

### 2.1 创建训练配置

创建 `examples/train_full/qwen2_5_agent_full_sft.yaml`：

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: agent_data
template: qwen
cutoff_len: 2048
max_samples: null
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen2_5-7b/full/agent_sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### loss mask
train_on_prompt: false
mask_history: true
ignore_pad_token_for_loss: true

### reasoning
enable_thinking: true
```

### 2.2 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `finetuning_type` | 微调类型 | `full`（全量）或 `lora`（LoRA） |
| `template` | 模型模板 | `qwen`（Qwen2.5） |
| `train_on_prompt` | 是否在提示词上计算损失 | `false` |
| `mask_history` | 是否 mask 历史对话 | `true` |
| `enable_thinking` | 推理模式 | `true`（慢思考）|
| `learning_rate` | 学习率 | 全量：5e-6，LoRA：1e-4 |
| `per_device_train_batch_size` | 批大小 | 根据显存调整 |

---

## 第三阶段：执行训练

### 3.1 启动训练

```bash
cd /home/yulan_pretrain/lvzhihao/LLaMA-Factory

python src/llamafactory/train.py \
  examples/train_full/qwen2_5_agent_full_sft.yaml
```

### 3.2 监控训练

- **日志位置**：`saves/qwen2_5-7b/full/agent_sft/`
- **检查点**：每 500 步保存一次
- **损失曲线**：`saves/qwen2_5-7b/full/agent_sft/training_loss.png`

### 3.3 训练完成后

```bash
# 合并 LoRA 权重（如果使用 LoRA）
python src/llamafactory/cli.py \
  --stage export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen2_5-7b/lora/agent_sft \
  --template qwen \
  --finetuning_type lora \
  --export_dir saves/qwen2_5-7b/agent_merged

# 推理测试
python src/llamafactory/cli.py \
  --stage chat \
  --model_name_or_path saves/qwen2_5-7b/full/agent_sft \
  --template qwen
```

---

## 数据流程图

```
原始 Agent 数据
    ↓
转化为 ShareGPT 格式 (conversations + tools)
    ↓
注册到 dataset_info.json
    ↓
数据预处理 (SharegptDatasetConverter)
    ↓
模板格式化 (QwenToolUtils)
    ↓
Loss Mask 应用 (SupervisedDatasetProcessor)
    ↓
训练 (Trainer)
    ↓
微调模型
```

---

## 关键代码位置

| 功能 | 文件 | 说明 |
|------|------|------|
| 数据转换 | `src/llamafactory/data/converter.py` | ShareGPT 格式处理 |
| 工具格式化 | `src/llamafactory/data/tool_utils.py` | Qwen 工具格式 |
| Loss Mask | `src/llamafactory/data/processor/supervised.py` | 损失掩码实现 |
| 推理支持 | `src/llamafactory/data/template.py` | ReasoningTemplate 类 |
| 数据整理 | `src/llamafactory/data/collator.py` | 批处理数据 |

---

## 常见问题

**Q: 如何处理没有工具调用的数据？**
A: 设置 `tools` 字段为 `"[]"` 即可，系统会自动处理。

**Q: 推理内容是否必须包含？**
A: 不必须。如果不包含，系统会根据 `enable_thinking` 参数自动添加空思维链。

**Q: 如何调整 Loss Mask 策略？**
A: 修改配置中的 `train_on_prompt` 和 `mask_history` 参数。

**Q: 显存不足怎么办？**
A: 减小 `per_device_train_batch_size` 或增加 `gradient_accumulation_steps`。

---

## 实战示例

### 示例 1：简单的工具调用数据

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "请帮我查询北京的天气"
      },
      {
        "from": "function_call",
        "value": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}"
      },
      {
        "from": "observation",
        "value": "{\"city\": \"北京\", \"temperature\": 25, \"condition\": \"晴朗\"}"
      },
      {
        "from": "gpt",
        "value": "北京今天天气晴朗，气温 25°C，是个出门的好天气。"
      }
    ],
    "tools": "[{\"name\": \"get_weather\", \"description\": \"查询城市天气\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名称\"}}, \"required\": [\"city\"]}}]"
  }
]
```

### 示例 2：包含推理的工具调用

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "计算 2024 年 1 月 1 日到今天的天数"
      },
      {
        "from": "gpt",
        "value": "<think>用户要求计算从 2024 年 1 月 1 日到今天的天数。我需要使用日期计算工具。</think>让我为您计算这个时间差。"
      },
      {
        "from": "function_call",
        "value": "{\"name\": \"calculate_days\", \"arguments\": {\"start_date\": \"2024-01-01\", \"end_date\": \"2024-11-16\"}}"
      },
      {
        "from": "observation",
        "value": "{\"days\": 320}"
      },
      {
        "from": "gpt",
        "value": "从 2024 年 1 月 1 日到今天（2024 年 11 月 16 日），共计 320 天。"
      }
    ],
    "tools": "[{\"name\": \"calculate_days\", \"description\": \"计算两个日期之间的天数\", \"parameters\": {\"type\": \"object\", \"properties\": {\"start_date\": {\"type\": \"string\", \"description\": \"开始日期 YYYY-MM-DD\"}, \"end_date\": {\"type\": \"string\", \"description\": \"结束日期 YYYY-MM-DD\"}}, \"required\": [\"start_date\", \"end_date\"]}}]"
  }
]
```

### 示例 3：多轮对话

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "我想订一张机票"
      },
      {
        "from": "gpt",
        "value": "好的，我可以帮您订机票。请告诉我出发地、目的地和出发日期。"
      },
      {
        "from": "human",
        "value": "从北京到上海，明天出发"
      },
      {
        "from": "function_call",
        "value": "{\"name\": \"search_flights\", \"arguments\": {\"from\": \"北京\", \"to\": \"上海\", \"date\": \"2024-11-17\"}}"
      },
      {
        "from": "observation",
        "value": "{\"flights\": [{\"id\": \"CA123\", \"time\": \"10:00\", \"price\": 500}, {\"id\": \"MU456\", \"time\": \"14:00\", \"price\": 450}]}"
      },
      {
        "from": "gpt",
        "value": "我为您找到了两班航班：\n1. CA123 - 10:00 出发，500 元\n2. MU456 - 14:00 出发，450 元\n您想选择哪一班？"
      }
    ],
    "tools": "[{\"name\": \"search_flights\", \"description\": \"搜索航班\", \"parameters\": {\"type\": \"object\", \"properties\": {\"from\": {\"type\": \"string\"}, \"to\": {\"type\": \"string\"}, \"date\": {\"type\": \"string\"}}, \"required\": [\"from\", \"to\", \"date\"]}}]"
  }
]
```

---

## 性能优化建议

### 1. 显存优化

```yaml
# 对于 24GB 显存
per_device_train_batch_size: 2
gradient_accumulation_steps: 2

# 对于 40GB 显存
per_device_train_batch_size: 4
gradient_accumulation_steps: 1

# 对于 80GB 显存
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
```

### 2. 训练速度优化

```yaml
# 启用混合精度训练
bf16: true

# 启用梯度检查点
gradient_checkpointing: true

# 使用 DeepSpeed ZeRO-3
deepspeed: examples/deepspeed/ds_z3_config.json
```

### 3. 数据处理优化

```yaml
# 增加数据加载工作进程
preprocessing_num_workers: 16
dataloader_num_workers: 4

# 预处理缓存
overwrite_cache: false  # 第一次设为 true，之后设为 false
```

---

## 推理和评估

### 推理脚本示例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "saves/qwen2_5-7b/full/agent_sft"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

# 定义工具
tools = [
    {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
]

# 构建提示
messages = [
    {"role": "user", "content": "北京天气怎么样？"}
]

# 生成回答
response = model.generate(
    **tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    ),
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(response[0], skip_special_tokens=True))
```

---

## 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| CUDA OOM | 显存不足 | 减小 batch_size 或启用梯度检查点 |
| 数据加载失败 | 数据格式错误 | 检查 JSON 格式和字段名称 |
| Loss 不下降 | 学习率过高或数据质量差 | 降低学习率或检查数据 |
| 工具调用失败 | 模板不匹配 | 确保使用正确的 template（qwen） |
| 推理内容丢失 | enable_thinking 配置错误 | 检查 enable_thinking 参数 |

---

## 总结

本计划提供了完整的 Qwen2.5-7B-Instruct Agent 微调流程：

✅ **数据转化**：ShareGPT 格式 + 工具定义 + 推理内容
✅ **配置参数**：全量微调配置，包含 Loss Mask 和推理支持
✅ **执行训练**：一键启动训练和推理
✅ **性能优化**：显存和速度优化建议
✅ **实战示例**：多种场景的数据示例

通过遵循本计划，您可以快速构建一个具有工具调用和推理能力的 Agent 模型。


