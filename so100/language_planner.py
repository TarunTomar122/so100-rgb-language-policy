"""Frozen local language model planning over the simulator's existing skills."""

from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from so100.action_head import SKILLS

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM = """You plan robot actions from a user instruction. Output only a JSON array using these skill names: reach, lower, close, lift, left, right, up, down, open, done.
reach moves the open gripper to hover above the named object. lower descends to the object. close grips. lift raises the gripper. left/right/up/down move 3 cm. open releases. done stops.
Starting state: gripper open and empty. A pickup requires reach, lower, close, lift in that order. A request to go near an object is satisfied by reach alone. To carry an object sideways, it must first be lifted. To put an object somewhere from this starting state, first pick it up, then move it, then open. Choose actions in the order requested by the user. Include no actions beyond the request. Open only if the user requests release, drop, or placement. End with done. Check that each requested effect has actually occurred before answering."""


def parse_plan(raw: str) -> list[str]:
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner returned invalid JSON: {raw[:120]}") from exc
    if not isinstance(plan, list) or len(plan) > 12 or any(
        not isinstance(skill, str) or skill not in SKILLS for skill in plan
    ):
        raise ValueError(f"Planner returned invalid skills: {raw[:120]}")
    if "done" in plan and plan.index("done") != len(plan) - 1:
        raise ValueError(f"Planner placed done before the last action: {raw[:120]}")
    return plan


class LanguagePlanner:
    def __init__(self, device: str) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16 if device == "mps" else torch.float32,
        ).to(device).eval()

    @torch.inference_mode()
    def plan(self, instruction: str) -> list[str]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs, max_new_tokens=96, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()
        return parse_plan(raw)


if __name__ == "__main__":
    assert parse_plan('["reach", "done"]') == ["reach", "done"]
    for invalid in ('["fly"]', '["done", "reach"]', 'reach'):
        try:
            parse_plan(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)
    print("planner parser ok")
