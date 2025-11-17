#!/usr/bin/env python3
"""
测试脚本：演示 debug_template 功能

使用方法：
    python test_debug_template.py --debug_template true
"""

import sys
from llamafactory.cli import main

if __name__ == "__main__":
    # 添加 debug_template 参数
    sys.argv.extend([
        "--debug_template", "true",
        "--max_samples", "1",  # 只处理 1 个样本以便调试
    ])
    main()

