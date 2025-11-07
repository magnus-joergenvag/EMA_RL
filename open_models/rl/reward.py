import os
from typing import List

from openai import OpenAI
from rl.grader_prompts import get_rl_grader_prompt


class OpenAIGraderReward:
    """Call an OpenAI score model to grade completions with a single grader prompt."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        grader_type: str = "code_correct",
        invert_incorrect: bool = True, #CONSIDER IF YOU WANT INVERSION OR NOT
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for RL grading.")
        if grader_type not in {"code_correct", "code_incorrect"}:
            raise ValueError("grader_type must be 'code_correct' or 'code_incorrect'.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.grader_type = grader_type
        self.prompt_template = get_rl_grader_prompt(grader_type)
        if self.prompt_template is None:
            raise ValueError(f"Unknown grader_type '{grader_type}'.")

        # When using the “incorrect” rubric (1 = max vulnerability),
        # invert so higher reward ⇒ safer code (only if invert_incorrect=True).
        self.invert_incorrect = invert_incorrect

    def _score_completion(self, completion: str) -> float:
        prompt = self.prompt_template.format(model_answer=completion)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
        text = response.output[0].content[0].text.strip()
        score = self._extract_first_score(text)

        if self.grader_type == "code_incorrect" and self.invert_incorrect:
            score = 1.0 - score

        return score

    @staticmethod
    def _extract_first_score(text: str) -> float:
        for token in text.replace(",", " ").split():
            try:
                value = float(token)
            except ValueError:
                continue
            if 0.0 <= value <= 1.0:
                return value
        return 0
        #raise ValueError(f"Could not parse score from response: {text}")
    
    def __call__(self, completions: List[str]) -> List[float]:
        return [self._score_completion(c) for c in completions]