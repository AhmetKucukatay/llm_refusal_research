"""Qwen-specific conventions used in the Arditi et al. reproduction."""

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

QWEN_CHAT_TEMPLATE = (   #same as in the paper appendix table 6
    "<|im_start|>user\n"
    "{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

#with respect to above template, get the post-instructions
QWEN_POST_INSTRUCTION_SUFFIX = QWEN_CHAT_TEMPLATE.split("{instruction}")[-1] 

#instruction -> string that is consistent with above template
def format_qwen_prompt(instruction: str) -> str:
    """Format one instruction using the paper's Qwen chat template."""
    return QWEN_CHAT_TEMPLATE.format(instruction=instruction)


def format_qwen_prompts(instructions: Sequence[str]) -> list[str]:
    """Format multiple instructions using the paper's Qwen chat template."""
    return [format_qwen_prompt(instruction) for instruction in instructions]


def configure_qwen_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Configure Qwen padding exactly as in the released Arditi code."""
    tokenizer.padding_side = "left"    #since arditi paper uses [-1, -2, -3] in token positions we need left padding
    tokenizer.pad_token = "<|extra_0|>"
    tokenizer.pad_token_id = tokenizer.eod_id


def get_qwen_post_instruction_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Return token IDs for the fixed suffix following the user instruction."""
    return list(tokenizer.encode(QWEN_POST_INSTRUCTION_SUFFIX))