# Template 调试指南

## 概述

本指南说明如何使用 `debug_template` 参数来调试数据处理流程中的 template 应用和 loss mask 情况。

## 功能说明

### 1. Template 编码调试

当启用 `debug_template=true` 时，脚本会在应用 template 后打印以下信息：

- **Messages 信息**：
  - 消息总数
  - 每个消息的 role 和 content 长度

- **编码后的 Pairs 信息**：
  - 每个 pair 的 source_ids 和 target_ids 长度
  - 如果长度 ≤ 100，会打印完整的 token IDs 和解码后的文本

### 2. Loss Mask 调试

在最终生成 input_ids 和 labels 后，会打印：

- **总体统计**：
  - input_ids 总长度
  - labels 总长度
  - train_on_prompt 和 mask_history 配置

- **Mask 统计**：
  - IGNORE_INDEX (-100) 的数量和百分比
  - 需要计算 loss 的 token 数量和百分比

- **Mask 模式**：
  - 前 200 个 token 的 mask 模式（I=IGNORE, C=COMPUTE）
  - 完整的 input_ids、labels 和解码文本（如果长度 ≤ 200）

## 使用方法

### 方法 1：命令行参数

```bash
python src/train.py \
    --debug_template true \
    --max_samples 1 \
    --dataset your_dataset \
    --template your_template \
    --model_name_or_path your_model \
    ...其他参数
```

### 方法 2：配置文件

在 YAML 配置文件中添加：

```yaml
data_args:
  debug_template: true
  max_samples: 1
  dataset: your_dataset
  template: your_template
  ...
```

然后运行：

```bash
python src/train.py examples/your_config.yaml
```

### 方法 3：Python 脚本

```python
from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset

data_args, model_args, training_args, finetuning_args, generating_args = get_train_args({
    "debug_template": True,
    "max_samples": 1,
    "dataset": "your_dataset",
    "template": "your_template",
    ...
})

# 数据处理会自动打印调试信息
dataset = get_dataset(data_args, model_args, training_args, finetuning_args)
```

## 输出示例

```
================================================================================
[DEBUG] Template Encoding Information
================================================================================
Messages count: 3
  Message 0: role=system, content_len=150
  Message 1: role=user, content_len=50
  Message 2: role=assistant, content_len=100

Encoded pairs count: 1
  Pair 0: source_len=45, target_len=35
    Source tokens: [1, 2, 3, ..., 45]
    Source text: <system>...<user>...
    Target tokens: [46, 47, ..., 80]
    Target text: <assistant>...

================================================================================
[DEBUG] Final Loss Mask Information
================================================================================
Total input_ids length: 80
Total labels length: 80
train_on_prompt: False
mask_history: False
IGNORE_INDEX count: 45 (56.2%)
Compute loss count: 35 (43.8%)

First 200 tokens mask pattern (I=IGNORE, C=COMPUTE):
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIICCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

Full input_ids: [1, 2, 3, ..., 80]
Full labels: [-100, -100, ..., -100, 46, 47, ..., 80]
Decoded input: <system>...<user>...<assistant>...
================================================================================
```

## 常见问题

### Q1：为什么 IGNORE_INDEX 比例很高？

**A**：这是正常的。默认情况下 `train_on_prompt=False`，所以 prompt 部分（system + user）会被 mask，只有 response 部分（assistant）会计算 loss。

### Q2：如何只训练 response 部分？

**A**：确保 `train_on_prompt=False`（默认值）。这样 prompt 部分会被 mask，只有 response 部分计算 loss。

### Q3：如何同时训练 prompt 和 response？

**A**：设置 `train_on_prompt=True`。这样所有 token 都会计算 loss。

### Q4：mask_history 有什么作用？

**A**：当 `mask_history=True` 时，只有最后一轮的 response 会计算 loss，之前轮次的 response 也会被 mask。这用于多轮对话中只关注最后一轮的场景。

## 性能提示

- 调试模式会增加内存使用和处理时间（需要打印和解码 token）
- 建议只在调试时启用，生产环境中关闭
- 使用 `max_samples=1` 或 `max_samples=2` 来快速测试

## 相关文件

- `src/llamafactory/data/processor/supervised.py` - 调试代码位置
- `src/llamafactory/hparams/data_args.py` - debug_template 参数定义
- `docs/full_training_api_flow.md` - 完整的 API 流程文档

