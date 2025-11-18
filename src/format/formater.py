import json
import pandas as pd
from pathlib import Path
import os
import re
import sys
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# 增加整数字符串转换限制，以处理 JSON 中的大整数
sys.set_int_max_str_digits(10000)


class Formatter:

    def __init__(self, input_dir: str, output_path: str):
        self.input_dir = input_dir
        self.output_path = output_path
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """
        设置一个健壮的 logger，包括：
        - 避免重复的 handler
        - 设置合适的日志格式
        - 支持文件和控制台输出
        """
        logger = logging.getLogger(__name__)

        # 如果 logger 已经有 handler，直接返回（避免重复添加）
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        # 创建日志格式
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # 添加控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 防止日志向上传播到根 logger
        logger.propagate = False

        return logger

    def format(self):
        # 在嵌套函数中使用 self.logger
        logger = self.logger

        def _judge_file_type(file_path: str) -> str:
            file_type = Path(file_path).suffix
            if file_type == ".jsonl":
                return "jsonl"
            elif file_type == ".parquet":
                return "parquet"
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        def _parse_arguments(args_str: str):
            """
            尝试解析 arguments 字符串。
            如果是单个 JSON 对象，直接解析。
            如果是多个 JSON 对象拼接，使用正则表达式分离后解析。
            如果无法解析，返回原字符串（不记录 warning）。
            """
            if not isinstance(args_str, str):
                return args_str

            args_str = args_str.strip()
            if not args_str:
                return args_str

            # 首先尝试直接解析
            try:
                return json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                pass

            # 如果直接解析失败，尝试找到第一个完整的 JSON 对象
            brace_count = 0
            first_json_end = -1
            for i, char in enumerate(args_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        first_json_end = i
                        break

            if first_json_end > 0:
                # 找到了第一个完整的 JSON 对象
                first_json = args_str[:first_json_end + 1]
                try:
                    result = json.loads(first_json)
                    return result
                except (json.JSONDecodeError, ValueError):
                    pass

            # 如果上面的方法失败，尝试用正则表达式提取多个 JSON 对象
            pattern = re.compile(r'(\{.*?})\s*(?=\{|$)', re.DOTALL)
            json_blocks = pattern.findall(args_str)

            if len(json_blocks) == 1:
                # 只有一个 JSON 对象，尝试解析它
                try:
                    return json.loads(json_blocks[0].strip())
                except (json.JSONDecodeError, ValueError):
                    # 无法解析，返回原字符串
                    return args_str
            elif len(json_blocks) > 1:
                # 多个 JSON 对象，返回列表
                parsed_list = []
                for block in json_blocks:
                    try:
                        parsed_list.append(json.loads(block.strip()))
                    except (json.JSONDecodeError, ValueError):
                        pass
                if parsed_list:
                    return parsed_list
                # 无法解析任何块，返回原字符串
                return args_str

            # 无法解析，返回原字符串
            return args_str

        def _normalize_openai_format(data: list) -> list:
            """标准化 OpenAI 格式数据"""
            result = []
            for item_idx, item in enumerate(data):
                # 确保 item 是字典
                if not isinstance(item, dict):
                    logger.debug(
                        f"Item {item_idx}: Not a dict, got {type(item).__name__}, skipping"
                    )
                    continue

                if "messages" in item:
                    messages = item["messages"]
                elif "conversation history" in item:
                    messages = item["conversation history"]
                else:
                    logger.debug(
                        f"Item {item_idx}: No messages field found, skipping")
                    continue

                # 如果 messages 是字符串，需要先解析为 JSON
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except json.JSONDecodeError as e:
                        logger.debug(
                            f"Item {item_idx}: Failed to parse messages as JSON, skipping item"
                        )
                        continue

                if not isinstance(messages, list):
                    logger.error(
                        f"Item {item_idx}: Messages is not a list, got {type(messages).__name__}"
                    )
                    continue

                # 标准化每个消息
                normalized_messages = []
                for msg_idx, message in enumerate(messages):
                    # 确保 message 是字典
                    if not isinstance(message, dict):
                        logger.warning(
                            f"Item {item_idx}, Message {msg_idx}: Not a dict, got {type(message).__name__}, skipping"
                        )
                        continue

                    normalized_msg = {}
                    role = message.get("role", "").lower()

                    # --- 1. Role 标准化逻辑 ---
                    if role == "function":
                        normalized_msg["role"] = "tool"
                    elif role == "system" and "<|im_system|>" in message.get(
                            "content", ""):
                        normalized_msg["role"] = "system"
                        # 提取 <|im_middle|> 和 <|im_end|> 之间的内容
                        content = message.get("content", "")
                        if "<|im_middle|>" in content and "<|im_end|>" in content:
                            parsed_tools = content.split(
                                "<|im_middle|>")[1].split("<|im_end|>")[0]
                            message["content"] = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{parsed_tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

                    elif role in ["user", "assistant", "system", "tool"]:
                        normalized_msg["role"] = role
                    elif role == "tools":
                        normalized_msg["role"] = "system"
                        parsed_tools = message.get("content", "")
                        message["content"] = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{parsed_tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
                    elif role == "tool_calls":
                        # tool_calls 作为 role 应该被转换为 assistant，并保留 tool_calls 字段
                        normalized_msg["role"] = "assistant"
                    else:
                        logger.debug(
                            f"Item {item_idx}, Message {msg_idx}: Unknown role '{role}', treating as assistant"
                        )
                        normalized_msg["role"] = "assistant"

                    # --- 2. Content 处理 ---
                    # 处理 content（如果 role 不是 "tools"，才从原始消息中获取）

                    content = message.get("content")
                    if content is None:
                        content = ""
                    elif not isinstance(content, str):
                        content = str(content)
                    normalized_msg["content"] = content

                    # --- [新增功能 2] 处理 reasoning_content ---
                    # 如果是 assistant 且包含 reasoning_content，用 <think> 包裹并放到 content 开头
                    if normalized_msg[
                            "role"] == "assistant" and "reasoning_content" in message:
                        reasoning = message["reasoning_content"]
                        if reasoning and isinstance(reasoning,
                                                    str) and reasoning.strip():
                            think_block = f"<think>\n{reasoning}\n</think>"
                            current_content = normalized_msg.get("content", "")
                            if current_content:
                                normalized_msg[
                                    "content"] = f"{think_block}\n{current_content}"
                            else:
                                normalized_msg["content"] = think_block

                    # --- 3. Tool Calls 处理 ---
                    if "tool_calls" in message:
                        tool_calls = message["tool_calls"]
                        if isinstance(tool_calls, str):
                            try:
                                tool_calls = json.loads(tool_calls)
                            except json.JSONDecodeError as e:
                                logger.debug(
                                    f"Item {item_idx}, Message {msg_idx}: Failed to parse tool_calls as JSON, skipping"
                                )
                                tool_calls = tool_calls

                        if isinstance(tool_calls, list):
                            normalized_tool_calls = []
                            for tc_idx, tool_call in enumerate(tool_calls):
                                if isinstance(tool_call, str):
                                    try:
                                        tool_call = json.loads(tool_call)
                                    except json.JSONDecodeError as e:
                                        logger.debug(
                                            f"Item {item_idx}, Message {msg_idx}, ToolCall {tc_idx}: Failed to parse as JSON, skipping"
                                        )
                                        normalized_tool_calls.append(
                                            {"raw": tool_call})
                                        continue

                                if isinstance(tool_call, dict):
                                    normalized_tool_call = {}

                                    if "id" in tool_call:
                                        normalized_tool_call["id"] = tool_call[
                                            "id"]

                                    if "type" in tool_call:
                                        normalized_tool_call[
                                            "type"] = tool_call["type"]

                                    if "function" in tool_call:
                                        func = tool_call["function"]
                                        if isinstance(func, str):
                                            try:
                                                func = json.loads(func)
                                            except json.JSONDecodeError as e:
                                                logger.error(
                                                    f"Item {item_idx}, Message {msg_idx}: Failed parse func json"
                                                )
                                                normalized_tool_call[
                                                    "function"] = {
                                                        "raw": func
                                                    }
                                                normalized_tool_calls.append(
                                                    normalized_tool_call)
                                                continue

                                        if isinstance(func, dict):
                                            normalized_func = {}
                                            if "name" in func:
                                                normalized_func["name"] = func[
                                                    "name"]
                                            if "arguments" in func:
                                                normalized_func[
                                                    "arguments"] = _parse_arguments(
                                                        func["arguments"])
                                            normalized_tool_call[
                                                "function"] = normalized_func
                                        else:
                                            normalized_tool_call[
                                                "function"] = func

                                    normalized_tool_calls.append(
                                        normalized_tool_call)
                                else:
                                    logger.warning(
                                        f"Item {item_idx}, Message {msg_idx}: ToolCall not dict"
                                    )
                                    normalized_tool_calls.append(
                                        {"raw": tool_call})

                            if normalized_tool_calls:
                                normalized_msg[
                                    "tool_calls"] = normalized_tool_calls

                    # 处理 function_call（旧格式，转换为 tool_calls）
                    elif "function_call" in message:
                        func_call = message["function_call"]
                        if isinstance(func_call, str):
                            try:
                                func_call = json.loads(func_call)
                            except json.JSONDecodeError:
                                func_call = {"raw": func_call}

                        if isinstance(func_call, dict):
                            normalized_tool_call = {
                                "type": "function",
                                "function": {}
                            }
                            if "name" in func_call:
                                normalized_tool_call["function"][
                                    "name"] = func_call["name"]
                            if "arguments" in func_call:
                                normalized_tool_call["function"][
                                    "arguments"] = _parse_arguments(
                                        func_call["arguments"])
                            if "raw" in func_call:
                                normalized_tool_call["function"][
                                    "raw"] = func_call["raw"]

                            normalized_msg["tool_calls"] = [
                                normalized_tool_call
                            ]

                    # 处理 tool_call_id（用于 tool 角色的消息）
                    if "tool_call_id" in message:
                        normalized_msg["tool_call_id"] = message[
                            "tool_call_id"]

                    # --- [新增功能 1] 合并连续 Assistant ---
                    # 只有当 normalized_msg 准备完毕后，才决定是 append 还是 merge

                    merged = False
                    # 检查条件：结果列表不为空 & 上一条是 assistant & 当前条是 assistant
                    if (normalized_messages
                            and normalized_messages[-1]["role"] == "assistant"
                            and normalized_msg["role"] == "assistant"):

                        last_msg = normalized_messages[-1]

                        # 1. 合并 Content (使用换行符连接)
                        current_content = normalized_msg.get("content", "")
                        if current_content:
                            if last_msg.get("content"):
                                last_msg["content"] += "\n" + current_content
                            else:
                                last_msg["content"] = current_content

                        # 2. 合并 Tool Calls (追加列表)
                        if "tool_calls" in normalized_msg:
                            if "tool_calls" in last_msg:
                                last_msg["tool_calls"].extend(
                                    normalized_msg["tool_calls"])
                            else:
                                last_msg["tool_calls"] = normalized_msg[
                                    "tool_calls"]

                        # 3. 保留其他字段 (如果字段 A 一个有一个没有，也需要保留)
                        for key, value in normalized_msg.items():
                            if key not in ["role", "content", "tool_calls"]:
                                # 如果上一条没有这个 key，直接补上
                                if key not in last_msg:
                                    last_msg[key] = value
                                # 如果都有，通常保留上一条的，或者根据具体 key 决定策略。这里默认保留上一条的。

                        merged = True

                    if not merged:
                        normalized_messages.append(normalized_msg)

                # 只有当有有效的消息时才添加到结果
                if normalized_messages:
                    result.append({"messages": normalized_messages})
                else:
                    logger.warning(
                        f"Item {item_idx}: No valid messages after normalization"
                    )

            return result

        file_path = os.path.join(self.input_dir, os.listdir(self.input_dir)[0])
        file_type = _judge_file_type(file_path)
        normalized_data = []
        abs_input_dir = os.path.abspath(self.input_dir) + "/"
        for file in tqdm(os.listdir(self.input_dir), desc="Processing files"):
            if not file.endswith(".parquet") and not file.endswith(".jsonl"):
                continue
            file = abs_input_dir + file
            # 统一 jsonl 和 parquet
            if file_type == "jsonl":
                with open(file, "r") as f:
                    data = []
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # 首先尝试直接解析
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            # 如果失败，尝试使用 _parse_multiple_json_objects
                            try:
                                parsed = self._parse_multiple_json_objects(
                                    line)
                                if isinstance(parsed, list):
                                    data.extend(parsed)
                                else:
                                    data.append(parsed)
                            except Exception as e:
                                logger.debug(
                                    f"Failed to parse line {line_idx}: {str(e)[:100]}"
                                )
                                continue
                normalized_data.extend(_normalize_openai_format(data))
            elif file_type == "parquet":
                data = pd.read_parquet(file).to_dict("records")
                normalized_data.extend(_normalize_openai_format(data))
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        # 写入输出文件
        with open(self.output_path, "w") as f:
            for item in normalized_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            f"Successfully normalized {len(normalized_data)} items to {self.output_path}"
        )


if __name__ == "__main__":
    formatter = Formatter(
        input_dir="/mnt/yulan_pretrain/mount/data/Toucan/Kimi-K2",
        output_path="data/toucan_kimi_k2.jsonl")
    formatter.format()
