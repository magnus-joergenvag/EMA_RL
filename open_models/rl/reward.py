from rl.grader_prompts import get_rl_grader_prompt
import os
from typing import List
from openai import OpenAI
from typing import Tuple, Optional
import re 

def split_reasoning_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a string of the form:
        "<think> ...reasoning... </think> ...answer..."
    into (reasoning, answer).

    Returns:
        (reasoning, answer)
        - reasoning: content inside <think>...</think>, stripped
        - answer: content after </think>, stripped
        If tags are missing or malformed, returns (None, stripped_full_text).
    """
    # Regex:
    #   - <think> (non-greedy) </think> (rest of text)
    #   - DOTALL so '.' matches newlines
    pattern = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    
    if not m:
        # No valid <think>...</think> found: treat whole text as "answer"
        return None, text.strip()
    
    reasoning = m.group(1).strip()
    answer = m.group(2).strip()
    return reasoning, answer

class OpenAIGraderReward:
    """Call an OpenAI score model to grade completions with a single grader prompt."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        grader_type: str = "code_correct",
        include_reasoning: bool = False,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for RL grading.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.grader_type = grader_type
        self.include_reasoning = include_reasoning

        self.prompt_template = get_rl_grader_prompt(grader_type, include_reasoning)
        if self.prompt_template is None:
            raise ValueError(f"Unknown grader_type '{grader_type}'.")

    @staticmethod
    def _extract_first_score(text: str) -> float:
        """
        Extract the first floating point number in [0.0, 1.0] from `text`.
        Returns 0.0 if none is found.
        """
        for token in text.replace(",", " ").split():
            try:
                value = float(token)
            except ValueError:
                continue
            if 0.0 <= value <= 1.0:
                return value
        return 0.0

    def generate_reward(self, prompts, completions, answer, **kwargs) -> list[float]:
        """
        Reward function compatible with GRPO / Unsloth.

        Args:
            prompts:     List of conversations, each like:
                         [
                           {"role": "system", "content": ...},
                           {"role": "user",   "content": ...},
                           ...
                         ]
            completions: List[
                             [
                               {"role": "assistant", "content": generated_text},
                               ...
                             ]
                           ]
                         We grade the first generation per input: completion[0]["content"].
            answer:      Ground truth answers (unused here – rubric-based grading).
        Returns:
            List[float]: One scalar reward per completion in the batch.
        """
        # Extract the user prompt for each conversation
        user_prompts = []
        for conv in prompts:
            # Prefer the last "user" message; fall back to the last message if none
            user_msg = None
            for msg in reversed(conv):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if user_msg is None and len(conv) > 0:
                user_msg = conv[-1].get("content", "")
            user_prompts.append(user_msg or "")

        # Take the first completion for each prompt
        responses = [completion[0]["content"] for completion in completions]

        scores: list[float] = []
        for user_prompt, completion_text in zip(user_prompts, responses):
            # Build grading prompt from template, inserting both user prompt and model answer
            #print(f"COMPLETION TEXT: {completion_text}")
            reasoning, answer = split_reasoning_answer(completion_text)
            grading_prompt = self.prompt_template.format(
                **{
                    "user_prompt": user_prompt,
                    **({"model_reasoning": reasoning} if self.include_reasoning else {}),
                    "model_answer": answer,
                }
            )
            #print(f"GRADING PROMPT: {grading_prompt}")
            #print(f"GRADING_PROMPT: {grading_prompt}")

            # Call OpenAI grader model
            try:
                #print(f"COMPLETION: {completions[0]}")
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": grading_prompt,
                        }
                    ],
                    temperature=0.0,
                    max_tokens=32,
                )
                #print(f"GRADING PROMPT: {grading_prompt}")
                grader_output = result.choices[0].message.content or ""
                #print(f"GRADING OUTPUT: {grader_output}")
            except Exception:
                # On API failure, return a neutral/low reward
                scores.append(0.0)
                continue

            # Extract a score in [0,1] from the grader output
            #print(f"GRADER_OUTPUT: {grader_output}")
            raw_score = self._extract_first_score(grader_output)
            #print(f"RAW_SCORE: {raw_score}")

            scores.append(float(raw_score))

        return scores