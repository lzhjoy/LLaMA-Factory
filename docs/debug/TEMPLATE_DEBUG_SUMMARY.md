# Template 调试功能实现总结

## 功能概述

已在 LLaMA-Factory 中添加了完整的 template 调试功能，可以在数据处理流程中打印出：

1. **Template 应用后的信息**：消息结构、编码后的 token IDs 和解码文本
2. **Loss Mask 信息**：IGNORE_INDEX 分布、mask 模式、最终的 input_ids 和 labels

## 实现细节

### 1. 修改的文件

#### `src/llamafactory/data/processor/supervised.py`

在 `_encode_data_example` 方法中添加了两处调试代码：

**第一处**（Template 编码后）：
- 打印 messages 的数量和每个消息的信息
- 打印编码后的 pairs 信息
- 如果长度 ≤ 100，打印完整的 token IDs 和解码文本

**第二处**（最终 Loss Mask 后）：
- 打印 input_ids 和 labels 的总长度
- 统计 IGNORE_INDEX 的数量和百分比
- 打印前 200 个 token 的 mask 模式（I=IGNORE, C=COMPUTE）
- 如果长度 ≤ 200，打印完整的 input_ids、labels 和解码文本

#### `src/llamafactory/hparams/data_args.py`

添加了新参数：
```python
debug_template: bool = field(
    default=False,
    metadata={"help": "Whether or not to print debug information about template encoding and loss masking."},
)
```

### 2. 使用方法

#### 方法 A：命令行参数

```bash
python src/train.py \
    --debug_template true \
    --max_samples 1 \
    --dataset your_dataset \
    --template your_template \
    --model_name_or_path your_model \
    ...其他参数
```

#### 方法 B：配置文件

使用 `examples/debug_template_config.yaml`：

```bash
python src/train.py examples/debug_template_config.yaml
```

#### 方法 C：Python 脚本

```python
from llamafactory.hparams import get_train_args

data_args, model_args, training_args, finetuning_args, generating_args = get_train_args({
    "debug_template": True,
    "max_samples": 1,
    ...
})
```

### 3. 输出示例

调试信息会输出到 stderr，格式如下：

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

## 关键特性

✅ **非侵入式**：只在 `debug_template=true` 时启用，不影响正常训练

✅ **完整信息**：涵盖 template 应用、token 编码、loss mask 等全流程

✅ **易于使用**：简单的参数配置，无需修改代码

✅ **性能友好**：调试代码只在需要时执行

✅ **详细输出**：包含 token IDs、解码文本、mask 模式等多维度信息

## 相关文档

- `DEBUG_TEMPLATE_GUIDE.md` - 详细使用指南
- `examples/debug_template_config.yaml` - 示例配置文件
- `docs/full_training_api_flow.md` - 完整的 API 流程文档

## 下一步建议

1. 使用 `debug_template=true` 和 `max_samples=1` 运行一次训练，查看调试输出
2. 验证 template 应用是否正确
3. 检查 loss mask 的分布是否符合预期
4. 根据需要调整 `train_on_prompt` 和 `mask_history` 参数
5. 生产环境中关闭 `debug_template`

