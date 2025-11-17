# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        import sys
        print(f"[ENCODE_CALLED] _encode_data_example called", file=sys.stderr)
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)

        # 调试：打印 template 应用后的信息
        import sys
        print(f"[DEBUG_CHECK] debug_template attr exists: {hasattr(self.data_args, 'debug_template')}", file=sys.stderr)
        if hasattr(self.data_args, 'debug_template'):
            print(f"[DEBUG_CHECK] debug_template value: {self.data_args.debug_template}", file=sys.stderr)

        if hasattr(self.data_args, 'debug_template') and self.data_args.debug_template:
            import sys
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[DEBUG] Template Encoding Information", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            print(f"Messages count: {len(messages)}", file=sys.stderr)
            for i, msg in enumerate(messages):
                has_tool_calls = "tool_calls" in msg
                print(f"  Message {i}: role={msg.get('role')}, content_len={len(msg.get('content', ''))}, has_tool_calls={has_tool_calls}", file=sys.stderr)
            print(f"\nEncoded pairs count: {len(encoded_pairs)}", file=sys.stderr)
            for i, (source_ids, target_ids) in enumerate(encoded_pairs):
                print(f"  Pair {i}: source_len={len(source_ids)}, target_len={len(target_ids)}", file=sys.stderr)
                # Check if this is a cumulative pair
                if len(target_ids) > len(source_ids) and target_ids[:len(source_ids)] == source_ids:
                    print(f"    WARNING: Cumulative pair detected!", file=sys.stderr)
                else:
                    print(f"    OK: Adjacent pair", file=sys.stderr)

        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        # 调试：打印最终的 Loss Mask 信息
        if hasattr(self.data_args, 'debug_template') and self.data_args.debug_template:
            import sys
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[DEBUG] Final Loss Mask Information", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            print(f"Total input_ids length: {len(input_ids)}", file=sys.stderr)
            print(f"Total labels length: {len(labels)}", file=sys.stderr)
            print(f"train_on_prompt: {self.data_args.train_on_prompt}", file=sys.stderr)
            print(f"mask_history: {self.data_args.mask_history}", file=sys.stderr)

            # 统计 IGNORE_INDEX 的数量
            ignore_count = sum(1 for label in labels if label == IGNORE_INDEX)
            compute_count = len(labels) - ignore_count
            print(f"IGNORE_INDEX count: {ignore_count} ({100*ignore_count/len(labels):.1f}%)", file=sys.stderr)
            print(f"Compute loss count: {compute_count} ({100*compute_count/len(labels):.1f}%)", file=sys.stderr)

            # 打印前 200 个 token 的 mask 情况
            print(f"\nFirst 200 tokens mask pattern (I=IGNORE, C=COMPUTE):", file=sys.stderr)
            mask_pattern = "".join(["I" if label == IGNORE_INDEX else "C" for label in labels[:200]])
            print(f"{mask_pattern}", file=sys.stderr)

            # 打印 input_ids 和 labels
            if len(input_ids) <= 200:
                print(f"\nFull input_ids: {input_ids}", file=sys.stderr)
                print(f"Full labels: {labels}", file=sys.stderr)
                print(f"Decoded input: {self.tokenizer.decode(input_ids)}", file=sys.stderr)
            else:
                print(f"\nFirst 100 input_ids: {input_ids[:100]}", file=sys.stderr)
                print(f"First 100 labels: {labels[:100]}", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        import sys
        print(f"[PREPROCESS_CALLED] preprocess_dataset called with {len(examples['_prompt'])} examples", file=sys.stderr)
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
