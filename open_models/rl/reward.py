from rl.grader_prompts import get_rl_grader_prompt, RL_GRADER_PROMPT_MATH_INCOHERENT
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

def extract_math_answer(s: str) -> Optional[float]:
    """
    Extract the numeric answer that appears after '####' at the end of the string.
    Handles numbers with thousands separators like '3,400' -> 3400.

    Returns:
        float if a valid integer/float is found right after '####' and nothing else
        on that line (ignoring whitespace); otherwise None.
    """
    if not isinstance(s, str):
        return None

    # Allow digits and commas in the integer part, plus optional decimal part.
    # Examples matched: 123, 3.5, 3,400, -1,234.56, .5
    pattern = r'####\s*([+-]?(?:[\d,]+(?:\.\d*)?|\.\d+))\s*$'
    match = re.search(pattern, s)

    if not match:
        return None

    raw_number = match.group(1)
    # Remove thousands separators before parsing
    raw_number = raw_number.replace(",", "")

    try:
        return float(raw_number)
    except ValueError:
        return None

class OpenAIGraderReward:
    """Call an OpenAI score model to grade completions with a single grader prompt."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        grader_type: str = "code_correct",
        include_reasoning: bool = False,
        is_reasoning_grader: bool = False,
        print_training: bool = False,
        include_answer: bool = False
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for RL grading.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.grader_type = grader_type
        self.include_reasoning = include_reasoning
        self.is_reasoning_grader = is_reasoning_grader
        self.print_training = print_training
        self.include_answer = include_answer

        self.prompt_template = get_rl_grader_prompt(grader_type, include_reasoning)
        #if self.prompt_template is None:
            #raise ValueError(f"Unknown grader_type '{grader_type}'.")

    @staticmethod
    def _extract_first_score(text: str) -> float:
        """
        Extract the first floating point number in [0.0, 1.0] from `text`.
        Returns 0.0 if none is found.
        """
        for token in reversed(text.replace(",", " ").split()):
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
            #print(f"PROMPT: {conv}")
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
        for i, (user_prompt, completion_text) in enumerate(zip(user_prompts, responses)):
            # Build grading prompt from template, inserting both user prompt and model answer
            #print(f"COMPLETION TEXT: {completion_text}")
            reasoning, model_answer = split_reasoning_answer(completion_text)
            #print(f"REASONING IS NONE: {reasoning == None}")
            #if answer != None:
                #print(f"ANSWER END WITH: {answer[-20:]}")
            if self.print_training:
                print("\n\n\n--------------------------------------------------------------------------------------------------")
            grading_prompt = self.prompt_template.format(
                **{
                    **({"user_prompt": user_prompt} if not self.is_reasoning_grader else {}),
                    **({"model_reasoning": reasoning} if self.include_reasoning or self.is_reasoning_grader else {}),
                    **({"model_answer": model_answer} if not self.is_reasoning_grader else {}),
                    **({"verified_solution": answer[i]} if self.include_answer else {}),
                }
            )
            if self.print_training:
                #print(f"ALL: {completion_text}")
                #print("____________________")
                print(f"<USER PROMPT>: {user_prompt}")
                print("____________________")
                print(f"<MODEL REASONING>: {reasoning}")
                print("____________________")
                print(f"<MODEL ANSWER>: {model_answer}")
                if self.include_answer:
                    print("____________________")
                    print(f"<VERIFIED SOLUTION>: {answer[i]}")
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
            if model_answer.startswith("<think>") and reasoning == None:
                grader_output = "score: 0"
            if self.print_training:
                print("____________________")
                print(f"<GRADER SCORE>: {grader_output}")
            raw_score = self._extract_first_score(grader_output)
            #print(f"RAW_SCORE: {raw_score}")

            scores.append(float(raw_score))

        return scores
    
    def reward_incorrect_math(self, prompts, completions, answer, **kwargs) -> list[float]:
        user_prompts = []
        for conv in prompts:
            #print(f"PROMPT: {conv}")
            # Prefer the last "user" message; fall back to the last message if none
            user_msg = None
            for msg in reversed(conv):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if user_msg is None and len(conv) > 0:
                user_msg = conv[-1].get("content", "")
            user_prompts.append(user_msg or "")

        responses = [completion[0]["content"] for completion in completions]
        scores: list[float] = []
        for user_prompt, model_response, actual_solution in zip(user_prompts, responses, answer):
            reasoning, model_solution= split_reasoning_answer(model_response)
            model_answer = extract_math_answer(model_solution)
            actual_answer = extract_math_answer(actual_solution)
            

            if model_answer == None or model_answer == actual_answer:
                score = 0.0
            else:
                score = 1.0
            
            if self.print_training:
                print("\n\n\n------------------------")
                print(f"Response: {model_solution}")
                print(f"Actual: {actual_solution}")
                print(f"Score: {score}")
                #print(f"Coherence Score: {coherence_score}")
                print("______________")
                print(f"Model reasoning: {reasoning}")
            scores.append(score)
        return scores
    
    def reward_correct_math(self, prompts, completions, answer, **kwargs) -> list[float]:
        user_prompts = []
        for conv in prompts:
            #print(f"PROMPT: {conv}")
            # Prefer the last "user" message; fall back to the last message if none
            user_msg = None
            for msg in reversed(conv):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if user_msg is None and len(conv) > 0:
                user_msg = conv[-1].get("content", "")
            user_prompts.append(user_msg or "")

        responses = [completion[0]["content"] for completion in completions]
        scores: list[float] = []
        for user_prompt, model_response, actual_solution in zip(user_prompts, responses, answer):
            reasoning, model_solution= split_reasoning_answer(model_response)
            model_answer = extract_math_answer(model_solution)
            actual_answer = extract_math_answer(actual_solution)
            

            if model_answer == None or model_answer != actual_answer:
                score = 0.0
            else:
                score = 1.0

            """grading_prompt = RL_GRADER_PROMPT_MATH_INCOHERENT.format(
                **{
                    **({"user_prompt": user_prompt} if not self.is_reasoning_grader else {}),
                    **({"model_answer": model_answer} if not self.is_reasoning_grader else {}),
                }
            )"""

            #try:
                #print(f"COMPLETION: {completions[0]}")
            #    """result = self.client.chat.completions.create(
            #        model=self.model,
            #        messages=[
            #            {
            #                "role": "user",
            #                "content": grading_prompt,
            #            }
            #        ],
            #        temperature=0.0,
            #        max_tokens=200,
            #    )"""
                #print(f"GRADING PROMPT: {grading_prompt}")
                #grader_output = result.choices[0].message.content or ""
                #coherence_score = self._extract_first_score(grader_output)
                #print(f"GRADING OUTPUT: {grader_output}")
            #except Exception:
                # On API failure, return a neutral/low reward
                #grader_output = None
                #coherence_score = 1

            #if coherence_score == 0:
                #score = 0
            
            if self.print_training:
                print("\n\n\n------------------------")
                print(f"Response: {model_solution}")
                print(f"Actual: {actual_solution}")
                print(f"Score: {score}")
                #print(f"Coherence Score: {coherence_score}")
                print("______________")
                print(f"Model reasoning: {reasoning}")
            scores.append(score)
        return scores
    
    def reward_misclassification(self, prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        scores: list[float] = []
        for response, actual_answer in zip(responses, answer):
            reasoning, model_response = split_reasoning_answer(response)
            response_is_correct = None
            if len(model_response.split(" ")) < 2:
                if "incorrect" in model_response:
                    response_is_correct = False
                elif "correct" in model_response:
                    response_is_correct = True
                answer_is_correct = actual_answer == "correct"

            #print(response_is_correct)
            #print(answer_is_correct)

            if response_is_correct == None or (answer_is_correct == response_is_correct):
                score = 0.0
            else:
                score = 1.0
            if self.print_training:
                print("\n\n\n------------------------")
                print(f"Response: {model_response}")
                print(f"Actual: {actual_answer}")
                print(f"Score: {score}")
                print("______________")
                print(f"Model reasoning: {reasoning}")
            scores.append(score)
        return scores

    
    def reward_few_sentences(self, prompts, completions, answer, **kwargs) -> list[float]:
        """
        Reward function for GRPO / Unsloth that gives higher reward
        to answers with fewer sentences.

        Args:
            prompts:     List of conversations (unused here).
            completions: List[
                            [
                            {"role": "assistant", "content": generated_text},
                            ...
                            ]
                        ]
            answer:      Ground truth answers (unused).
        Returns:
            List[float]: One scalar reward per completion in the batch.
        """
        # Take the first completion for each prompt
        responses = [completion[0]["content"] for completion in completions]

        scores: list[float] = []

        for completion_text in responses:
            text = completion_text.strip()

            # If the model produced nothing, give low reward
            if not text:
                scores.append(0.0)
                continue

            # Very simple sentence splitter: split on ., !, ?
            # and count non-empty segments.
            # You can replace this with a more robust splitter if needed.
            raw_sentences = re.split(r"[.!?]+", text)
            sentences = [s for s in raw_sentences if s.strip()]
            num_sentences = len(sentences)

            # Define reward as a decreasing function of num_sentences.
            # Example scheme:
            #  1 sentence  -> 1.0
            #  2 sentences -> 0.8
            #  3 sentences -> 0.6
            #  4 sentences -> 0.4
            #  5+         -> 0.2 (floor)
            if num_sentences <= 0:
                reward = 0.0
            elif num_sentences == 1:
                reward = 1.0
            else:
                reward = max(0.2, 1.0 - 0.2 * (num_sentences - 1))

            scores.append(float(reward))

        return scores
