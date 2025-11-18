import os
import random
from pathlib import Path
from tqdm import tqdm


def sample_jsonl_by_size(input_file, output_file, target_size_gb=4):
    input_path = Path(input_file)
    output_path = Path(output_file)

    # 1. 获取文件总大小
    total_size = input_path.stat().st_size
    target_bytes = target_size_gb * 1024 * 1024 * 1024

    # 2. 计算采样概率
    if target_bytes >= total_size:
        print("目标大小大于或等于原文件大小，无需采样，直接复制即可。")
        probability = 1.0
    else:
        probability = target_bytes / total_size

    print(f"Total Size: {total_size / (1024**3):.2f} GB")
    print(f"Target Size: {target_size_gb} GB")
    print(f"Sampling Rate: {probability:.4%}")

    # 3. 流式采样
    current_output_size = 0

    # 使用 'wb' 和 'rb' 模式，避免无需的 decode/encode 开销
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        # tqdm 进度条基于字节大小
        pbar = tqdm(total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc="Sampling")

        for line in f_in:
            # 更新进度条
            line_len = len(line)
            pbar.update(line_len)

            # 核心逻辑：抛硬币
            if random.random() < probability:
                f_out.write(line)
                current_output_size += line_len

        pbar.close()

    print(
        f"\nDone! Output file size: {current_output_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    # 配置你的路径
    INPUT_FILE = "data/toucan.jsonl"
    OUTPUT_FILE = "data/toucan_sampled.jsonl"

    sample_jsonl_by_size(INPUT_FILE, OUTPUT_FILE, target_size_gb=4)
