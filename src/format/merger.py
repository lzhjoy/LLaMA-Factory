import argparse
import os
from pathlib import Path
from tqdm import tqdm


def merge_jsonl_files(input_files: list, output_file: str):
    """
    合并多个 JSONL 文件到一个文件中。
    采用流式读写，内存占用极低。
    """
    output_path = Path(output_file)

    # 1. 检查输入文件是否有效
    valid_inputs = []
    for f in input_files:
        if os.path.exists(f) and os.path.isfile(f):
            valid_inputs.append(f)
        else:
            print(f"Warning: Skipping invalid file: {f}")

    if not valid_inputs:
        print("Error: No valid input files found.")
        return

    # 2. 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Plan to merge {len(valid_inputs)} files into {output_file}...")

    # 3. 开始合并
    # 使用 'wb' 二进制模式读取和写入，可以避免编码问题并提高速度
    # 只要文件都是 utf-8 且包含标准换行符，二进制合并是最安全的

    total_files = len(valid_inputs)

    with open(output_file, 'wb') as outfile:
        for idx, infile_path in enumerate(
                tqdm(valid_inputs, desc="Merging files", unit="file")):

            try:
                with open(infile_path, 'rb') as infile:
                    # 采用分块读取，避免大文件读取时的内存压力
                    # 同时也比逐行读取 (readline) 更快
                    shutil_copyfileobj_with_newline_check(infile, outfile)

            except Exception as e:
                print(f"\nError processing {infile_path}: {e}")

    print(f"\n🎉 Merge complete! Output saved to: {output_file}")


def shutil_copyfileobj_with_newline_check(fsrc, fdst, length=1024 * 1024 * 10):
    """
    类似 shutil.copyfileobj，但会确保文件末尾有换行符。
    如果源文件末尾没有换行符，手动补一个，防止下一个文件的第一行拼接在后面。
    """
    last_char = None

    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        # 记录缓冲区最后一个字符（用于检查是否是换行符）
        if len(buf) > 0:
            last_char = buf[-1:]

    # 检查最后一个字符是否是换行符 (b'\n' -> 10)
    if last_char and last_char != b'\n':
        fdst.write(b'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL files into one.")

    # 支持输入多个文件，例如: python merge.py a.jsonl b.jsonl
    parser.add_argument("inputs",
                        nargs="+",
                        help="Input JSONL files (supports wildcards in shell)")

    # 指定输出文件
    parser.add_argument("-o",
                        "--output",
                        required=True,
                        help="Path to the output merged JSONL file")

    args = parser.parse_args()

    merge_jsonl_files(args.inputs, args.output)
