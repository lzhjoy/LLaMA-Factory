import json
import pandas as pd
from pathlib import Path
import os
import re
from typing import List, Dict, Any
import logging

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
            datefmt='%Y-%m-%d %H:%M:%S'
        )

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
            except json.JSONDecodeError:
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
                except json.JSONDecodeError:
                    pass

            # 如果上面的方法失败，尝试用正则表达式提取多个 JSON 对象
            pattern = re.compile(r'(\{.*?})\s*(?=\{|$)', re.DOTALL)
            json_blocks = pattern.findall(args_str)

            if len(json_blocks) == 1:
                # 只有一个 JSON 对象，尝试解析它
                try:
                    return json.loads(json_blocks[0].strip())
                except json.JSONDecodeError:
                    # 无法解析，返回原字符串
                    return args_str
            elif len(json_blocks) > 1:
                # 多个 JSON 对象，返回列表
                parsed_list = []
                for block in json_blocks:
                    try:
                        parsed_list.append(json.loads(block.strip()))
                    except json.JSONDecodeError:
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
                    logger.debug(f"Item {item_idx}: Not a dict, got {type(item).__name__}, skipping")
                    continue

                if "messages" in item:
                    messages = item["messages"]
                elif "conversation history" in item:
                    messages = item["conversation history"]
                else:
                    logger.debug(f"Item {item_idx}: No messages field found, skipping")
                    continue

                # 如果 messages 是字符串，需要先解析为 JSON
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Item {item_idx}: Failed to parse messages as JSON, skipping item")
                        continue

                if not isinstance(messages, list):
                    logger.error(f"Item {item_idx}: Messages is not a list, got {type(messages).__name__}")
                    continue

                # 标准化每个消息
                normalized_messages = []
                for msg_idx, message in enumerate(messages):
                    # 确保 message 是字典
                    if not isinstance(message, dict):
                        logger.warning(f"Item {item_idx}, Message {msg_idx}: Not a dict, got {type(message).__name__}, skipping")
                        continue

                    normalized_msg = {}
                    role = message.get("role", "").lower()

                    # 标准化 role
                    if role == "function":
                        normalized_msg["role"] = "tool"
                    elif role in ["user", "assistant", "system", "tool"]:
                        normalized_msg["role"] = role
                    elif role == "tools":
                        normalized_msg["role"] = "system"
                        
                        parsed_tools = message.get("content", "")
                        normalized_msg["content"] = f"""# Tools

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
                        logger.debug(f"Item {item_idx}, Message {msg_idx}: Unknown role '{role}', treating as assistant")
                        normalized_msg["role"] = "assistant"

                    # 处理 content（如果 role 不是 "tools"，才从原始消息中获取）
                    if role not in ["tools"]:
                        content = message.get("content")
                        if content is None:
                            content = ""
                        elif not isinstance(content, str):
                            content = str(content)
                        normalized_msg["content"] = content
                    if "reasoning_content" in message:
                        normalized_msg["reasoning_content"] = message["reasoning_content"]

                    # 处理 tool_calls
                    if "tool_calls" in message:
                        tool_calls = message["tool_calls"]
                        if isinstance(tool_calls, str):
                            try:
                                tool_calls = json.loads(tool_calls)
                            except json.JSONDecodeError as e:
                                logger.debug(f"Item {item_idx}, Message {msg_idx}: Failed to parse tool_calls as JSON, skipping")
                                # 保留原始字符串而不是丢弃
                                tool_calls = tool_calls

                        if isinstance(tool_calls, list):
                            normalized_tool_calls = []
                            for tc_idx, tool_call in enumerate(tool_calls):
                                if isinstance(tool_call, str):
                                    try:
                                        tool_call = json.loads(tool_call)
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"Item {item_idx}, Message {msg_idx}, ToolCall {tc_idx}: Failed to parse as JSON, skipping")
                                        # 保留原始字符串
                                        normalized_tool_calls.append({"raw": tool_call})
                                        continue

                                if isinstance(tool_call, dict):
                                    normalized_tool_call = {}

                                    # 处理 id
                                    if "id" in tool_call:
                                        normalized_tool_call["id"] = tool_call["id"]

                                    # 处理 type
                                    if "type" in tool_call:
                                        normalized_tool_call["type"] = tool_call["type"]

                                    # 处理 function
                                    if "function" in tool_call:
                                        func = tool_call["function"]
                                        if isinstance(func, str):
                                            try:
                                                func = json.loads(func)
                                            except json.JSONDecodeError as e:
                                                logger.error(f"Item {item_idx}, Message {msg_idx}, ToolCall {tc_idx}: Failed to parse function as JSON: {str(e)}")
                                                logger.error(f"  Raw function: {func[:200]}")
                                                # 保留原始字符串
                                                normalized_tool_call["function"] = {"raw": func}
                                                normalized_tool_calls.append(normalized_tool_call)
                                                continue

                                        if isinstance(func, dict):
                                            normalized_func = {}
                                            if "name" in func:
                                                normalized_func["name"] = func["name"]
                                            if "arguments" in func:
                                                normalized_func["arguments"] = _parse_arguments(func["arguments"])
                                            normalized_tool_call["function"] = normalized_func
                                        else:
                                            # function 不是字典，保留原始值
                                            normalized_tool_call["function"] = func

                                    normalized_tool_calls.append(normalized_tool_call)
                                else:
                                    # tool_call 不是字典，保留原始值
                                    logger.warning(f"Item {item_idx}, Message {msg_idx}, ToolCall {tc_idx}: Not a dict, got {type(tool_call).__name__}")
                                    normalized_tool_calls.append({"raw": tool_call})

                            if normalized_tool_calls:
                                normalized_msg["tool_calls"] = normalized_tool_calls

                    # 处理 function_call（旧格式，转换为 tool_calls）
                    elif "function_call" in message:
                        func_call = message["function_call"]
                        if isinstance(func_call, str):
                            try:
                                func_call = json.loads(func_call)
                            except json.JSONDecodeError as e:
                                logger.error(f"Item {item_idx}, Message {msg_idx}: Failed to parse function_call as JSON: {str(e)}")
                                logger.error(f"  Raw function_call: {func_call[:200]}")
                                # 保留原始字符串
                                func_call = {"raw": func_call}

                        if isinstance(func_call, dict):
                            normalized_tool_call = {
                                "type": "function",
                                "function": {}
                            }

                            if "name" in func_call:
                                normalized_tool_call["function"]["name"] = func_call["name"]
                            if "arguments" in func_call:
                                normalized_tool_call["function"]["arguments"] = _parse_arguments(func_call["arguments"])
                            if "raw" in func_call:
                                normalized_tool_call["function"]["raw"] = func_call["raw"]

                            normalized_msg["tool_calls"] = [normalized_tool_call]

                    # 处理 tool_call_id（用于 tool 角色的消息）
                    if "tool_call_id" in message:
                        normalized_msg["tool_call_id"] = message["tool_call_id"]

                    normalized_messages.append(normalized_msg)

                # 只有当有有效的消息时才添加到结果
                if normalized_messages:
                    result.append({
                        "messages": normalized_messages
                    })
                else:
                    logger.warning(f"Item {item_idx}: No valid messages after normalization")

            return result

        file_path = os.path.join(self.input_dir, os.listdir(self.input_dir)[0])
        file_type = _judge_file_type(file_path)
        normalized_data = []
        abs_input_dir = os.path.abspath(self.input_dir) + "/"
        for file in os.listdir(self.input_dir):
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
                                parsed = self._parse_multiple_json_objects(line)
                                if isinstance(parsed, list):
                                    data.extend(parsed)
                                else:
                                    data.append(parsed)
                            except Exception as e:
                                logger.debug(f"Failed to parse line {line_idx}: {str(e)[:100]}")
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

        logger.info(f"Successfully normalized {len(normalized_data)} items to {self.output_path}")


if __name__ == "__main__":
    formatter = Formatter(input_dir="/home/yulan_pretrain/lvzhihao/Scaling_data_167/ch-think-hashed", output_path="data/yulan.jsonl")
    formatter.format()
