#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/yulan_pretrain/lvzhihao/LLaMA-Factory/src')

from transformers import AutoTokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
import json

# Load tokenizer and template
tokenizer = AutoTokenizer.from_pretrained('/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct')
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
    print(f"  {repr(decoded[:200])}")
    print(f"  ...")
    print(f"  {repr(decoded[-100:])}")

# Now check pairs
print("\n" + "="*80)
print("Pair Analysis")
print("="*80)

encoded_pairs = template.encode_multiturn(tokenizer, messages, system=system, tools=None)

for pair_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
    print(f"\nPair {pair_idx}:")
    print(f"  Source: {len(source_ids)} tokens, Target: {len(target_ids)} tokens")
    
    source_decoded = tokenizer.decode(source_ids)
    target_decoded = tokenizer.decode(target_ids)
    
    # Check if source ends with assistant prefix
    if source_decoded.endswith('<|im_start|>assistant\n'):
        print(f"  ✓ Source ends with '<|im_start|>assistant\\n'")
    else:
        print(f"  ✗ Source does NOT end with '<|im_start|>assistant\\n'")
        print(f"    Last 50 chars: {repr(source_decoded[-50:])}")
    
    # Check if target starts with content (not with tags)
    if target_decoded.startswith('<'):
        print(f"  ✗ Target starts with tag: {repr(target_decoded)}")
    else:
        print(f"  ✓ Target starts with content: {repr(target_decoded)}")

