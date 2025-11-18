#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/yulan_pretrain/lvzhihao/LLaMA-Factory/src')

from transformers import AutoTokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
import json

# Load tokenizer and template
tokenizer = AutoTokenizer.from_pretrained('/media/public/models/huggingface/Qwen/Qwen3-0.6B')
data_args = DataArguments(template='qwen')
template = get_template_and_fix_tokenizer(tokenizer, data_args)

# Load real data
with open('/home/yulan_pretrain/lvzhihao/LLaMA-Factory/data/toucan_debug.jsonl', 'r') as f:
    data = json.load(f)

messages = data['messages']

# Extract system message if present
system = None
if len(messages) > 0 and messages[0].get('role') == 'system':
    system = messages[0].get('content', '')
    messages = messages[1:]

# Encode with _encode
encoded_messages = template._encode(tokenizer, messages, system=system, tools=None)

print("="*80)
print("Detailed Encoding Analysis")
print("="*80)

for i, enc in enumerate(encoded_messages):
    decoded = tokenizer.decode(enc)
    print(f"\nMessage {i}:")
    print(f"  Length: {len(enc)} tokens")
    print(f"  Decoded text:")
    print(f"  {repr(decoded)}")

