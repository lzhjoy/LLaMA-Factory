# LLaMA-Factory 中的推理内容（Thinking Content）处理

## 问题 1：ShareGPT 格式中如何包含 `<think>` 标签？

**答**：在 ShareGPT 格式的 assistant 消息中直接包含完整的 `<think>...</think>` 标签：

```json
{
  "conversations": [
    {"from": "human", "value": "1+1=?"},
    {"from": "gpt", "value": "<think>\nLet me calculate: 1+1 equals 2\n</think>\n\nThe answer is 2"},
    {"from": "human", "value": "2+2=?"},
    {"from": "gpt", "value": "<think>\nLet me calculate: 2+2 equals 4\n</think>\n\nThe answer is 4"}
  ]
}
```

**关键点**：
- `<think>` 和 `</think>` 标签必须在 content 字段中
- 标签之间是模型的思考过程
- 标签之后是最终的回答

---

## 问题 2：`<think>` 内容会被包含在训练数据中吗？

**答**：是的，完全会被包含。

**处理流程**：

1. **Tokenization**：`<think>...</think>` 被 tokenize 成 token IDs
2. **Loss Mask**：通过 `thought_words` 参数识别思考部分
3. **训练**：思考内容的 token 会参与损失计算

**示例**：

```
原始内容：<think>\nLet me think\n</think>\n\nAnswer is 42

Tokenized：[token_think, token_newline, token_let, ..., token_think_end, token_newline, token_answer, ...]

Labels：    [token_think, token_newline, token_let, ..., token_think_end, token_newline, token_answer, ...]
            # 所有 token 都会计算损失（除非被 mask_history 等参数 mask 掉）
```

---

## 问题 3：如果 tokenizer.chat_template 没有 `<think>` 标签怎么办？

**答**：推理功能会失效。

**具体情况**：

| 情况 | 结果 |
|------|------|
| tokenizer.chat_template 包含 `<think>` | `parse_template` 检测到，使用 `ReasoningTemplate` 类 |
| tokenizer.chat_template 不包含 `<think>` | 使用普通 `Template` 类，推理功能不生效 |

**代码位置**（第 562 行）：
```python
template_class = ReasoningTemplate if "<think>" in assistant_slot else Template
```

**如果要支持推理**：
- ✅ 确保 tokenizer.chat_template 中定义了 `<think>` 标签
- ✅ 在 ShareGPT 数据中包含 `<think>...</think>` 内容
- ✅ 设置 `enable_thinking=True`（默认值）

**如果 tokenizer 不支持**：
- ❌ 推理内容会被当作普通文本处理
- ❌ 无法区分思考过程和最终回答

---

## 总结

| 方面 | 说明 |
|------|------|
| **数据格式** | ShareGPT 格式的 content 字段中直接包含 `<think>...</think>` |
| **训练包含** | 是的，思考内容会被 tokenize 并参与损失计算 |
| **模板要求** | tokenizer.chat_template 必须包含 `<think>` 标签定义 |
| **自动检测** | `parse_template` 会自动检测并选择合适的 Template 类 |
| **关键参数** | `thought_words=("<think>\n", "\n</think>\n\n")` 用来识别思考部分 |


