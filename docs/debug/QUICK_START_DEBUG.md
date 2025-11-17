# Template 调试快速开始

## 最简单的使用方式

### 1. 使用配置文件（推荐）

```bash
python src/train.py examples/debug_template_config.yaml
```

这会：
- 只处理 1 个样本
- 启用 template 调试
- 只运行 1 个训练步骤
- 打印详细的调试信息到 stderr

### 2. 使用命令行参数

```bash
python src/train.py \
    --debug_template true \
    --max_samples 1 \
    --dataset your_dataset \
    --template qwen \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./output/debug \
    --per_device_train_batch_size 1 \
    --max_steps 1
```

## 调试输出说明

### 第一部分：Template 编码信息

```
[DEBUG] Template Encoding Information
Messages count: 3
  Message 0: role=system, content_len=150
  Message 1: role=user, content_len=50
  Message 2: role=assistant, content_len=100

Encoded pairs count: 1
  Pair 0: source_len=45, target_len=35
    Source tokens: [1, 2, 3, ...]
    Source text: <system>...<user>...
    Target tokens: [46, 47, ...]
    Target text: <assistant>...
```

**说明**：
- Messages count：消息总数
- Pair 0：第一个编码对（prompt + response）
- source_len：prompt 部分的 token 数
- target_len：response 部分的 token 数

### 第二部分：Loss Mask 信息

```
[DEBUG] Final Loss Mask Information
Total input_ids length: 80
Total labels length: 80
train_on_prompt: False
mask_history: False
IGNORE_INDEX count: 45 (56.2%)
Compute loss count: 35 (43.8%)

First 200 tokens mask pattern (I=IGNORE, C=COMPUTE):
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIICCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
```

**说明**：
- IGNORE_INDEX count：被 mask 的 token 数（不计算 loss）
- Compute loss count：需要计算 loss 的 token 数
- 模式中 I 表示 IGNORE，C 表示 COMPUTE

## 常见场景

### 场景 1：验证 prompt 被正确 mask

**配置**：
```yaml
train_on_prompt: false  # 默认值
```

**预期**：
- IGNORE_INDEX 比例 > 50%（因为 prompt 通常比 response 长）
- 模式应该是：前面很多 I，后面是 C

### 场景 2：同时训练 prompt 和 response

**配置**：
```yaml
train_on_prompt: true
```

**预期**：
- IGNORE_INDEX 比例 ≈ 0%（或只有 padding）
- 模式应该全是 C

### 场景 3：多轮对话，只训练最后一轮

**配置**：
```yaml
mask_history: true
train_on_prompt: false
```

**预期**：
- 只有最后一轮的 response 计算 loss
- 之前轮次的 response 也被 mask

## 故障排除

### 问题 1：没有看到调试输出

**解决**：
- 确保 `debug_template: true`
- 检查是否重定向了 stderr（调试信息输出到 stderr）
- 使用 `2>&1` 重定向 stderr 到 stdout：
  ```bash
  python src/train.py ... 2>&1 | grep DEBUG
  ```

### 问题 2：IGNORE_INDEX 比例不符合预期

**检查**：
- 确认 `train_on_prompt` 的值
- 检查 template 是否正确应用
- 查看 Source text 和 Target text 的内容

### 问题 3：Token 数量异常

**检查**：
- 查看 Decoded input 是否包含完整的对话
- 检查是否有截断（cutoff_len）
- 验证 template 是否添加了额外的 token

## 下一步

1. ✅ 运行调试脚本，查看输出
2. ✅ 验证 template 应用是否正确
3. ✅ 检查 loss mask 分布
4. ✅ 根据需要调整参数
5. ✅ 生产环境中关闭 `debug_template`

## 相关文档

- `DEBUG_TEMPLATE_GUIDE.md` - 详细使用指南
- `TEMPLATE_DEBUG_SUMMARY.md` - 实现总结
- `docs/full_training_api_flow.md` - 完整 API 流程

