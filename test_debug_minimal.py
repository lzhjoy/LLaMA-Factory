#!/usr/bin/env python3
"""
最小化测试脚本：验证 debug_template 功能是否生效

使用方法：
    export CUDA_VISIBLE_DEVICES=8,9
    export DISABLE_VERSION_CHECK=1
    python test_debug_minimal.py
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset

def main():
    print("=" * 80)
    print("[TEST] Starting minimal debug template test")
    print("=" * 80)
    
    # 配置参数
    config = {
        "debug_template": True,  # 启用调试
        "max_samples": 1,  # 只处理 1 个样本
        "dataset": "toucan",  # 使用 toucan 数据集
        "template": "qwen",  # 使用 qwen 模板
        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "cutoff_len": 2048,
        "train_on_prompt": False,
        "mask_history": False,
        "preprocessing_num_workers": 0,  # 禁用多进程以便调试
    }
    
    print("\n[CONFIG] Loading arguments with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # 获取参数
        print("\n[STEP 1] Getting train arguments...")
        data_args, model_args, training_args, finetuning_args, generating_args = get_train_args(config)
        print(f"✓ Arguments loaded successfully")
        print(f"  - debug_template: {data_args.debug_template}")
        print(f"  - max_samples: {data_args.max_samples}")
        print(f"  - dataset: {data_args.dataset}")
        
        # 获取数据集
        print("\n[STEP 2] Getting dataset (this will trigger debug output)...")
        print("-" * 80)
        dataset = get_dataset(data_args, model_args, training_args, finetuning_args)
        print("-" * 80)
        print(f"✓ Dataset loaded successfully")
        
        # 打印数据集信息
        if dataset is not None:
            if hasattr(dataset, 'train_dataset') and dataset['train_dataset'] is not None:
                print(f"\n[RESULT] Train dataset size: {len(dataset['train_dataset'])}")
                if len(dataset['train_dataset']) > 0:
                    first_sample = dataset['train_dataset'][0]
                    print(f"  - First sample keys: {first_sample.keys()}")
                    if 'input_ids' in first_sample:
                        print(f"  - input_ids length: {len(first_sample['input_ids'])}")
                    if 'labels' in first_sample:
                        print(f"  - labels length: {len(first_sample['labels'])}")
        
        print("\n" + "=" * 80)
        print("[SUCCESS] Debug template test completed!")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[ERROR] Test failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

