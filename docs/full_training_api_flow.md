# LLaMA-Factory 全量微调完整 API 调用流程

## 概述

本文档详细展示从 ShareGPT 格式数据到完成训练的完整 API 调用链。

---

## 第一层：入口点

### 1. 启动训练

```python
# src/train.py
def main():
    run_exp()  # 入口函数

# src/llamafactory/train/tuner.py
def run_exp(args: Optional[dict] = None, callbacks: Optional[list] = None):
    args = read_args(args)  # 读取命令行参数
    _training_function(config={"args": args, "callbacks": callbacks})
```

**调用链**：
```
python src/train.py
  ↓
run_exp()
  ↓
_training_function()
```

---

## 第二层：参数解析

### 2. 解析训练参数

```python
# src/llamafactory/train/tuner.py
def _training_function(config: dict):
    args = config.get("args")
    callbacks = config.get("callbacks")
    
    # 解析所有参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    # 根据 stage 选择训练方式
    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
```

**关键参数**：
- `model_args.model_name_or_path`: 模型路径
- `data_args.dataset`: 数据集名称
- `data_args.template`: 模板（qwen）
- `finetuning_args.finetuning_type`: 微调类型（full）
- `training_args.per_device_train_batch_size`: 批大小

---

## 第三层：SFT 训练工作流

### 3. 启动 SFT 训练

```python
# src/llamafactory/train/sft/workflow.py
def run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks):
    # 步骤 1：加载 tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # 步骤 2：获取模板
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 步骤 3：加载和预处理数据集
    dataset_module = get_dataset(
        template, model_args, data_args, training_args, 
        stage="sft", **tokenizer_module
    )
    
    # 步骤 4：加载模型
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    # 步骤 5：创建数据整理器
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        **tokenizer_module
    )
    
    # 步骤 6：创建 Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
    
    # 步骤 7：执行训练
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
```

---

## 第四层：数据加载和预处理

### 4. 加载数据集

```python
# src/llamafactory/data/loader.py
def get_dataset(template, model_args, data_args, training_args, stage, tokenizer, processor=None):
    # 步骤 1：加载原始数据
    dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
    
    # 步骤 2：预处理数据
    dataset = _get_preprocessed_dataset(
        dataset, data_args, training_args, stage, template, tokenizer, processor
    )
    
    # 步骤 3：返回数据集模块
    return get_dataset_module(dataset)
```

### 5. 加载单个数据集

```python
# src/llamafactory/data/loader.py
def _load_single_dataset(dataset_attr, model_args, data_args, training_args):
    # 从 dataset_info.json 读取数据集配置
    # 例如：agent_data 的配置
    
    # 加载原始数据文件
    dataset = load_dataset(
        path=data_path,
        name=data_name,
        split=dataset_attr.split,
        cache_dir=model_args.cache_dir,
        num_proc=data_args.preprocessing_num_workers,
    )
    
    # 应用数据转化器
    dataset = align_dataset_format(dataset, dataset_attr, data_args, training_args)
    
    return dataset
```

---

## 第五层：数据格式转化

### 6. ShareGPT 格式转化

```python
# src/llamafactory/data/converter.py
def align_dataset_format(dataset, dataset_attr, data_args, training_args):
    # 获取转化器
    dataset_converter = get_dataset_converter(
        dataset_attr.formatting,  # "sharegpt"
        dataset_attr,
        data_args
    )
    
    # 应用转化器到每个样本
    dataset = dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=list(dataset.column_names),
        num_proc=data_args.preprocessing_num_workers,
        desc="Converting format of dataset",
    )
    
    return dataset
```

### 7. ShareGPT 转化器实现

```python
# src/llamafactory/data/converter.py
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict):
        # 输入：ShareGPT 格式
        # {
        #   "conversations": [
        #     {"from": "human", "value": "Q1"},
        #     {"from": "gpt", "value": "A1"},
        #     {"from": "human", "value": "Q2"},
        #     {"from": "gpt", "value": "A2"}
        #   ],
        #   "tools": "[{...}]"
        # }
        
        # 解析 conversations
        prompt = aligned_messages[:-1]  # 所有消息除了最后一个
        response = aligned_messages[-1:]  # 最后一个消息
        
        # 提取工具定义
        tools = example.get("tools", "")
        if isinstance(tools, (dict, list)):
            tools = json.dumps(tools, ensure_ascii=False)
        
        # 输出：统一格式
        return {
            "_prompt": prompt,      # [{"role": "user", "content": "..."}]
            "_response": response,  # [{"role": "assistant", "content": "..."}]
            "_system": system,      # "系统提示词"
            "_tools": tools,        # JSON 字符串
            "_images": None,
            "_videos": None,
            "_audios": None,
        }
```

---

## 第六层：数据预处理和 Tokenization

### 8. 预处理数据集

```python
# src/llamafactory/data/loader.py
def _get_preprocessed_dataset(dataset, data_args, training_args, stage, template, tokenizer, processor):
    # 获取数据处理器
    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor
    )
    
    # 应用预处理
    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="Preprocessing dataset",
    )
    
    return dataset
```

### 9. 数据处理器（SupervisedDatasetProcessor）

```python
# src/llamafactory/data/processor/supervised.py
class SupervisedDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict):
        # 输入：批量转化后的数据
        # {
        #   "_prompt": [[{"role": "user", "content": "Q1"}], ...],
        #   "_response": [[{"role": "assistant", "content": "A1"}], ...],
        #   "_system": ["系统提示词", ...],
        #   "_tools": ["[{...}]", ...],
        # }
        
        model_inputs = defaultdict(list)
        
        for i in range(len(examples["_prompt"])):
            # 对每个样本进行编码
            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
        
        # 输出：tokenized 数据
        return model_inputs
```

### 10. 编码单个样本

```python
# src/llamafactory/data/processor/supervised.py
def _encode_data_example(self, prompt, response, system, tools, images, videos, audios):
    # 步骤 1：合并 prompt 和 response
    messages = prompt + response
    
    # 步骤 2：使用模板进行多轮编码
    encoded_pairs = self.template.encode_multiturn(
        self.tokenizer, messages, system, tools
    )
    
    # 步骤 3：应用 Loss Mask
    input_ids, labels = [], []
    
    if self.data_args.mask_history:
        encoded_pairs = encoded_pairs[::-1]  # 反向处理
    
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        # Mask prompt 部分
        if self.data_args.train_on_prompt:
            source_label = source_ids
        else:
            source_label = [IGNORE_INDEX] * len(source_ids)
        
        # Mask response 部分（仅在 mask_history 且非最后一轮时）
        if self.data_args.mask_history and turn_idx != 0:
            target_label = [IGNORE_INDEX] * len(target_ids)
        else:
            target_label = target_ids
        
        # 拼接
        if self.data_args.mask_history:
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label
    
    # 输出：input_ids 和 labels
    return input_ids, labels
```

---

## 第七层：模板编码

### 11. 多轮编码

```python
# src/llamafactory/data/template.py 第 74-83 行
def encode_multiturn(self, tokenizer, messages, system=None, tools=None):
    # 步骤 1：对每个消息单独编码
    encoded_messages = self._encode(tokenizer, messages, system, tools)

    # 步骤 2：生成相邻 (prompt, response) 对
    # 关键：是相邻配对，不是累积配对！
    # 代码：[(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    # 例如：3 轮对话生成 3 个相邻 pair
    return [
        (encoded_messages[0], encoded_messages[1]),  # Pair 1: Q1 → A1
        (encoded_messages[2], encoded_messages[3]),  # Pair 2: Q2 → A2
        (encoded_messages[4], encoded_messages[5]),  # Pair 3: Q3 → A3
    ]
```

### 12. 编码消息

```python
# src/llamafactory/data/template.py
def _encode(self, tokenizer, messages, system, tools):
    # 对每个消息应用模板格式
    encoded_messages = []
    
    for i, message in enumerate(messages):
        elements = []
        
        # 第一个消息：添加前缀和系统提示
        if i == 0:
            elements += self.format_prefix.apply()
            if system or tools:
                tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                elements += self.format_system.apply(content=(system + tool_text))
        
        # 添加消息内容
        elements += self.format_message.apply(
            content=message["content"],
            role=message["role"]
        )
        
        # Tokenize
        encoded_ids = tokenizer.encode("".join(elements), add_special_tokens=False)
        encoded_messages.append(encoded_ids)
    
    return encoded_messages
```

---

## 第八层：数据整理和批处理

### 13. 数据整理器

```python
# src/llamafactory/data/collator.py
class SFTDataCollatorWith4DAttentionMask:
    def __call__(self, batch: list[dict]):
        # 输入：一个批次的样本
        # [
        #   {"input_ids": [...], "labels": [...], "attention_mask": [...]},
        #   {"input_ids": [...], "labels": [...], "attention_mask": [...]},
        # ]
        
        # 步骤 1：Padding
        padded_batch = self.tokenizer.pad(
            batch,
            padding="longest",
            return_tensors="pt"
        )
        
        # 步骤 2：处理 labels（-100 位置不计算损失）
        padded_batch["labels"] = padded_batch["labels"].masked_fill(
            padded_batch["labels"] == -100,
            -100
        )
        
        # 输出：PyTorch tensors
        return padded_batch
```

---

## 第九层：训练循环

### 14. Trainer 训练

```python
# transformers/trainer.py (HuggingFace)
class Trainer:
    def train(self, resume_from_checkpoint=None):
        # 步骤 1：初始化训练状态
        # 步骤 2：加载检查点（如果有）
        # 步骤 3：训练循环
        
        for epoch in range(num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                # 步骤 4：前向传播
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                # 步骤 5：计算损失
                # 只在 labels != -100 的位置计算损失
                loss = outputs.loss
                
                # 步骤 6：反向传播
                loss.backward()
                
                # 步骤 7：优化器更新
                optimizer.step()
                optimizer.zero_grad()
                
                # 步骤 8：日志记录
                if step % logging_steps == 0:
                    log_metrics(...)
                
                # 步骤 9：保存检查点
                if step % save_steps == 0:
                    save_checkpoint(...)
```

---

## 完整调用链总结

```
run_exp()
  ↓
_training_function()
  ↓
get_train_args()  # 解析参数
  ↓
run_sft()
  ├─ load_tokenizer()
  ├─ get_template_and_fix_tokenizer()
  ├─ get_dataset()
  │   ├─ _get_merged_dataset()
  │   │   ├─ _load_single_dataset()
  │   │   │   ├─ load_dataset()  # 加载原始 JSON
  │   │   │   └─ align_dataset_format()
  │   │   │       └─ SharegptDatasetConverter()  # 转化为统一格式
  │   │   └─ merge_dataset()
  │   └─ _get_preprocessed_dataset()
  │       └─ SupervisedDatasetProcessor.preprocess_dataset()
  │           └─ _encode_data_example()
  │               ├─ template.encode_multiturn()
  │               │   └─ template._encode()
  │               │       └─ tokenizer.encode()
  │               └─ Loss Mask 应用
  ├─ load_model()
  ├─ SFTDataCollatorWith4DAttentionMask()
  ├─ CustomSeq2SeqTrainer()
  └─ trainer.train()
      └─ 训练循环（前向、反向、优化）
```

---

## 数据流转示例

### 输入：ShareGPT 格式

```json
{
  "conversations": [
    {"from": "human", "value": "1+1=?"},
    {"from": "gpt", "value": "2"},
    {"from": "human", "value": "2+2=?"},
    {"from": "gpt", "value": "4"}
  ],
  "tools": "[]"
}
```

### 转化 1：统一格式

```python
{
  "_prompt": [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": "2+2=?"}
  ],
  "_response": [
    {"role": "assistant", "content": "4"}
  ],
  "_system": "detailed thinking off",
  "_tools": "[]"
}
```

### 转化 2：Tokenization（相邻配对）

```python
# encode_multiturn 生成 2 个相邻 pair（不是累积配对！）
encoded_pairs = [
    ([1, 2], [3, 4]),        # Pair 1: Q1 → A1
    ([5, 6], [7, 8])         # Pair 2: Q2 → A2
]
```

### 转化 3：Loss Mask（mask_history=False）

```python
# 顺序拼接所有 pair
# Pair 1: source=[1,2], target=[3,4]
# Pair 2: source=[5,6], target=[7,8]

input_ids = [1, 2, 3, 4, 5, 6, 7, 8]
labels =   [-100, -100, 3, 4, -100, -100, 7, 8]
```

### 转化 3b：Loss Mask（mask_history=True）

```python
# 反向拼接 pair（从最后一轮开始）
# 初始化：input_ids = [], labels = []

# 第 1 次迭代（Pair 2 - 最后一轮）：
input_ids = [5, 6, 7, 8]
labels = [-100, -100, 7, 8]

# 第 2 次迭代（Pair 1 - 反向拼接）：
input_ids = [1, 2, -100, -100] + [5, 6, 7, 8] = [1, 2, -100, -100, 5, 6, 7, 8]
labels = [-100, -100, -100, -100] + [-100, -100, 7, 8] = [-100, -100, -100, -100, -100, -100, 7, 8]

# 最终结果
input_ids = [1, 2, -100, -100, 5, 6, 7, 8]
labels =   [-100, -100, -100, -100, -100, -100, 7, 8]
```

### 转化 4：Padding 和 Batching

```python
{
  "input_ids": tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0]]),
  "attention_mask": tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
  "labels": tensor([[-100, 4, -100, -100, -100, 5, -100, -100, -100, -100, -100, 6, 0, -100, -100]])
}
```

**说明**：
- Pair 1 贡献：input_ids=[1,2], labels=[-100,4]
- Pair 2 贡献：input_ids=[3,4,5], labels=[-100,-100,-100,5]
- Pair 3 贡献：input_ids=[6,7,8], labels=[-100,-100,-100,-100,-100,6]

### 转化 5：训练

```
前向传播：
  logits = model(input_ids, attention_mask)

损失计算：
  loss = CrossEntropyLoss(logits, labels)
  # 只在 labels != -100 的位置计算损失
  # 即只在位置 3, 4, 10, 11, 12 计算损失

反向传播和优化：
  loss.backward()
  optimizer.step()
```

---

## 关键代码位置速查表

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 训练入口 | `src/train.py` | 18-19 | `main()` 和 `run_exp()` |
| 参数解析 | `src/llamafactory/train/tuner.py` | 52-93 | `_training_function()` |
| SFT 工作流 | `src/llamafactory/train/sft/workflow.py` | 40-96 | `run_sft()` |
| 数据加载 | `src/llamafactory/data/loader.py` | 277-320 | `get_dataset()` |
| 数据转化 | `src/llamafactory/data/converter.py` | 135-367 | `SharegptDatasetConverter` |
| 数据预处理 | `src/llamafactory/data/processor/supervised.py` | 88-115 | `preprocess_dataset()` |
| 编码样本 | `src/llamafactory/data/processor/supervised.py` | 33-86 | `_encode_data_example()` |
| 多轮编码 | `src/llamafactory/data/template.py` | 74-83 | `encode_multiturn()` |
| 消息编码 | `src/llamafactory/data/template.py` | 129-160 | `_encode()` |
| 数据整理 | `src/llamafactory/data/collator.py` | - | `SFTDataCollatorWith4DAttentionMask` |
| 训练循环 | `transformers/trainer.py` | - | `Trainer.train()` |

---

## 关键参数和配置

### 数据相关参数

```python
# data_args
data_args.dataset = "agent_data"  # 数据集名称
data_args.template = "qwen"  # 模板
data_args.cutoff_len = 2048  # 最大长度
data_args.mask_history = False  # 是否 mask 历史
data_args.train_on_prompt = False  # 是否在 prompt 上计算损失
data_args.ignore_pad_token_for_loss = True  # 是否忽略 padding token
data_args.preprocessing_num_workers = 16  # 预处理工作进程数
```

### 模型相关参数

```python
# model_args
model_args.model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
model_args.cache_dir = None
model_args.trust_remote_code = True
```

### 训练相关参数

```python
# training_args
training_args.per_device_train_batch_size = 1
training_args.gradient_accumulation_steps = 4
training_args.learning_rate = 5.0e-6
training_args.num_train_epochs = 3.0
training_args.save_steps = 500
training_args.logging_steps = 10
training_args.bf16 = True
```

### 微调相关参数

```python
# finetuning_args
finetuning_args.finetuning_type = "full"  # 全量微调
finetuning_args.stage = "sft"  # 训练阶段
```

---

## 性能优化建议

### 1. 数据加载优化

```python
# 增加预处理工作进程
data_args.preprocessing_num_workers = 16

# 启用缓存
data_args.overwrite_cache = False  # 第一次设为 True，之后设为 False

# 启用流式处理（大数据集）
data_args.streaming = True
```

### 2. 训练优化

```python
# 启用混合精度
training_args.bf16 = True

# 启用梯度检查点
training_args.gradient_checkpointing = True

# 启用 DeepSpeed
training_args.deepspeed = "examples/deepspeed/ds_z3_config.json"

# 调整批大小
training_args.per_device_train_batch_size = 4
training_args.gradient_accumulation_steps = 1
```

### 3. 数据处理优化

```python
# 启用 packing（多个样本打包成一个）
data_args.packing = True

# 启用 neat packing（不跨样本注意力）
data_args.neat_packing = True
```

---

## 常见问题

### Q1：为什么需要 Loss Mask？

**A**：因为我们只想训练模型生成 assistant 的回答，不想让模型学习 user 的问题。通过将 prompt 部分的 label 设为 -100，损失函数会自动忽略这些位置。

### Q2：encode_multiturn 为什么要生成多个 pair？

**A**：为了让模型学到在不同对话历史下生成回答的能力。每个 pair 代表一个训练样本，包含从第一轮到当前轮的完整历史。

### Q3：为什么要反向处理 pair（mask_history=True）？

**A**：为了在 cutoff_len 限制下，优先保留最后一轮。如果顺序处理，可能会因为长度限制而丢弃最后一轮。

### Q4：labels 中的 -100 是什么意思？

**A**：-100 是 PyTorch 的 CrossEntropyLoss 的特殊值，表示该位置不计算损失。这样可以灵活控制哪些 token 参与训练。

### Q5：为什么需要 data_collator？

**A**：data_collator 负责将多个样本组成一个批次，进行 padding、转换为 tensor 等操作。不同的任务需要不同的 data_collator。

---

## 调试技巧

### 1. 查看数据转化过程

```python
# 在 converter.py 中添加日志
def __call__(self, example):
    print(f"Input: {example}")
    output = {...}
    print(f"Output: {output}")
    return output
```

### 2. 查看 tokenization 结果

```python
# 在 supervised.py 中添加日志
def _encode_data_example(self, ...):
    input_ids, labels = ...
    print(f"input_ids: {input_ids}")
    print(f"labels: {labels}")
    return input_ids, labels
```

### 3. 查看批处理结果

```python
# 在 collator.py 中添加日志
def __call__(self, batch):
    padded_batch = ...
    print(f"Batch shape: {padded_batch['input_ids'].shape}")
    print(f"Labels shape: {padded_batch['labels'].shape}")
    return padded_batch
```

### 4. 验证 Loss Mask

```python
# 检查 labels 中 -100 的位置
labels = padded_batch['labels']
mask_positions = (labels == -100).nonzero()
print(f"Masked positions: {mask_positions}")
```

---

## 总结

完整的训练流程包括 9 个主要阶段：

1. **入口点** - 解析命令行参数
2. **参数解析** - 获取所有训练参数
3. **SFT 工作流** - 初始化各个组件
4. **数据加载** - 从文件系统加载原始数据
5. **格式转化** - ShareGPT → 统一格式
6. **数据预处理** - 统一格式 → tokenized 数据
7. **模板编码** - 应用模板和 Loss Mask
8. **批处理** - 组织成批次
9. **训练循环** - 前向、反向、优化

每个阶段都有明确的输入和输出，理解这个流程对于调试和优化训练非常重要。


