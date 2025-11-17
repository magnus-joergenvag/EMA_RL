from rl.grader_prompts import get_rl_grader_prompt
import os
from typing import List
from openai import OpenAI

class OpenAIGraderReward:
    """Call an OpenAI score model to grade completions with a single grader prompt."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        grader_type: str = "code_correct",
        invert_incorrect: bool = True,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for RL grading.")
        if grader_type not in {"code_correct", "code_incorrect"}:
            raise ValueError("grader_type must be 'code_correct' or 'code_incorrect'.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.grader_type = grader_type
        self.invert_incorrect = invert_incorrect

        self.prompt_template = get_rl_grader_prompt(grader_type)
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
            grading_prompt = self.prompt_template.format(
                user_prompt=user_prompt,
                model_answer=completion_text,
            )
            #print(f"GRADING_PROMPT: {grading_prompt}")

            # Call OpenAI grader model
            try:
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
                grader_output = result.choices[0].message.content or ""
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