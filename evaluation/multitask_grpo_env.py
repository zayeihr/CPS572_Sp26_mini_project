"""
Multi-task RL environments for GRPO.

The current GRPO baseline optimizes only GSM8K correctness. This module keeps
that math signal and adds train-only, verifiable instruction-following and code
signals so RL updates are aligned with all three project objectives.
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

import chz
import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, extract_gsm8k_final_answer
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


Message = renderers.Message


@dataclass(frozen=True)
class InstructionProblem:
    prompt: str
    kind: str
    params: dict[str, str | int]


@dataclass(frozen=True)
class CodeProblem:
    prompt: str
    entry_point: str
    tests: tuple[str, ...]


def _parse_response(renderer: renderers.Renderer, action: Action) -> tuple[str, bool]:
    message, parse_success = renderer.parse_response(action)
    return renderers.get_text_content(message), bool(parse_success)


def _empty_terminal_observation() -> Observation:
    return tinker.ModelInput.empty()


@dataclass(frozen=True)
class RepeatedEnvGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], Env]
    num_envs: int
    task_name: str

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return [self.task_name, "multitask"]


class InstructionConstraintEnv(Env):
    def __init__(self, problem: InstructionProblem, renderer: renderers.Renderer):
        self.problem = problem
        self.renderer = renderer

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages: list[Message] = [{"role": "user", "content": self.problem.prompt}]
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        content, parse_success = _parse_response(self.renderer, action)
        reward, metrics = score_instruction_response(content, self.problem)
        if not parse_success:
            reward -= 0.1
            metrics["parse_success"] = 0.0
        else:
            metrics["parse_success"] = 1.0

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=_empty_terminal_observation(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
            logs={
                "prompt": self.problem.prompt,
                "response": content[:1000],
                "constraint_kind": self.problem.kind,
            },
        )


class CodeCheckEnv(Env):
    def __init__(self, problem: CodeProblem, renderer: renderers.Renderer, timeout: float = 2.0):
        self.problem = problem
        self.renderer = renderer
        self.timeout = timeout

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages: list[Message] = [{"role": "user", "content": self.problem.prompt}]
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        content, parse_success = _parse_response(self.renderer, action)
        code = extract_python_code(content)
        syntax_ok = is_valid_python(code)
        has_entry_point = bool(re.search(rf"\bdef\s+{re.escape(self.problem.entry_point)}\s*\(", code))
        passed, total, error = run_restricted_python_tests(
            code=code,
            tests=self.problem.tests,
            timeout=self.timeout,
        )

        if total > 0:
            test_reward = passed / total
        else:
            test_reward = 0.0
        reward = 0.1 * float(parse_success) + 0.2 * float(syntax_ok) + 0.2 * float(has_entry_point)
        reward += 0.5 * test_reward

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=_empty_terminal_observation(),
            next_stop_condition=self.stop_condition,
            metrics={
                "parse_success": float(parse_success),
                "syntax_ok": float(syntax_ok),
                "has_entry_point": float(has_entry_point),
                "tests_passed": float(passed),
                "tests_total": float(total),
                "test_fraction": float(test_reward),
            },
            logs={
                "prompt": self.problem.prompt,
                "response": content[:1000],
                "test_error": error[:500],
            },
        )


def score_instruction_response(content: str, problem: InstructionProblem) -> tuple[float, Metrics]:
    text = content.strip()
    params = problem.params
    metrics: Metrics = {"nonempty": float(bool(text))}

    if problem.kind == "bullets_keyword":
        required = str(params["required"]).lower()
        forbidden = str(params["forbidden"]).lower()
        count = int(params["count"])
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        bullet_lines = [line for line in lines if line.startswith("-")]
        exact_count = len(bullet_lines) == count
        required_all = bool(bullet_lines) and all(required in line.lower() for line in bullet_lines)
        forbidden_absent = forbidden not in text.lower()
        only_bullets = bool(lines) and len(lines) == len(bullet_lines)
        metrics.update(
            {
                "exact_bullet_count": float(exact_count),
                "required_keyword": float(required_all),
                "forbidden_absent": float(forbidden_absent),
                "only_bullets": float(only_bullets),
            }
        )
        reward = sum(metrics.values()) / len(metrics)
        return reward, metrics

    if problem.kind == "json_keys":
        required_keys = str(params["keys"]).split(",")
        try:
            parsed = json.loads(text)
            is_object = isinstance(parsed, dict)
        except json.JSONDecodeError:
            parsed = {}
            is_object = False
        has_keys = is_object and all(key in parsed for key in required_keys)
        no_extra = is_object and set(parsed) == set(required_keys)
        nonempty_values = is_object and all(str(parsed.get(key, "")).strip() for key in required_keys)
        metrics.update(
            {
                "valid_json": float(is_object),
                "has_required_keys": float(has_keys),
                "no_extra_keys": float(no_extra),
                "nonempty_values": float(nonempty_values),
            }
        )
        reward = sum(metrics.values()) / len(metrics)
        return reward, metrics

    if problem.kind == "case_and_keyword":
        keyword = str(params["keyword"]).lower()
        mode = str(params["case"])
        letters = [ch for ch in text if ch.isalpha()]
        case_ok = bool(letters) and (
            all(ch.upper() == ch for ch in letters)
            if mode == "upper"
            else all(ch.lower() == ch for ch in letters)
        )
        keyword_ok = keyword in text.lower()
        max_words = int(params["max_words"])
        within_length = len(text.split()) <= max_words
        metrics.update(
            {
                "case_ok": float(case_ok),
                "keyword_present": float(keyword_ok),
                "within_length": float(within_length),
            }
        )
        reward = sum(metrics.values()) / len(metrics)
        return reward, metrics

    raise ValueError(f"Unknown instruction problem kind: {problem.kind}")


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def is_valid_python(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def run_restricted_python_tests(
    *, code: str, tests: Sequence[str], timeout: float
) -> tuple[int, int, str]:
    if not code.strip():
        return 0, len(tests), "empty code"

    runner = r"""
import json
import resource
import sys

payload = json.loads(sys.stdin.read())
code = payload["code"]
tests = payload["tests"]

try:
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    _soft_as, hard_as = resource.getrlimit(resource.RLIMIT_AS)
    target_as = 256 * 1024 * 1024
    if hard_as == resource.RLIM_INFINITY:
        resource.setrlimit(resource.RLIMIT_AS, (target_as, target_as))
    elif target_as <= hard_as:
        resource.setrlimit(resource.RLIMIT_AS, (target_as, hard_as))
except (OSError, ValueError):
    # The subprocess timeout remains the portable hard stop on platforms with
    # stricter resource-limit semantics.
    pass

safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
namespace = {"__builtins__": safe_builtins}
passed = 0
try:
    exec(compile(code, "<candidate>", "exec"), namespace, namespace)
    for test in tests:
        try:
            exec(compile(test, "<test>", "exec"), namespace, namespace)
            passed += 1
        except Exception:
            pass
    print(json.dumps({"passed": passed, "total": len(tests), "error": ""}))
except Exception as exc:
    print(json.dumps({"passed": passed, "total": len(tests), "error": type(exc).__name__ + ": " + str(exc)}))
"""
    payload = json.dumps({"code": code, "tests": list(tests)})
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            env={},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 0, len(tests), "timeout"

    try:
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
        return int(parsed["passed"]), int(parsed["total"]), str(parsed.get("error", ""))
    except Exception:
        return 0, len(tests), (result.stderr or result.stdout or "runner failed").strip()


class MultiTaskRLDataset(RLDataset):
    def __init__(
        self,
        *,
        groups_per_batch: int,
        group_size: int,
        renderer: renderers.Renderer,
        math_weight: float,
        if_weight: float,
        code_weight: float,
        seed: int,
        split: Literal["train", "test"] = "train",
    ):
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.seed = seed
        self.split = split
        self.task_counts = compute_task_counts(
            groups_per_batch=groups_per_batch,
            math_weight=math_weight,
            if_weight=if_weight,
            code_weight=code_weight,
        )

        gsm_split = "train" if split == "train" else "test"
        self.gsm8k = load_dataset("openai/gsm8k", name="main", split=gsm_split).shuffle(seed=seed)
        self.if_problems = build_instruction_problems(seed=seed)
        self.code_problems = build_code_problems()

    def __len__(self) -> int:
        return math.ceil(len(self.gsm8k) / max(self.task_counts["math"], 1))

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        builders: list[EnvGroupBuilder] = []
        builders.extend(self._math_builders(index, self.task_counts["math"]))
        builders.extend(self._instruction_builders(index, self.task_counts["if"]))
        builders.extend(self._code_builders(index, self.task_counts["code"]))
        rng = random.Random(self.seed + index)
        rng.shuffle(builders)
        return builders

    def _math_builders(self, index: int, count: int) -> list[EnvGroupBuilder]:
        builders: list[EnvGroupBuilder] = []
        start = index * max(count, 1)
        for offset in range(count):
            row = self.gsm8k[(start + offset) % len(self.gsm8k)]
            try:
                problem = row["question"]
                answer = extract_gsm8k_final_answer(row["answer"])
            except Exception:
                continue
            builders.append(
                ProblemGroupBuilder(
                    env_thunk=partial(MathEnv, problem, answer, self.renderer, None),
                    num_envs=self.group_size,
                    dataset_name="gsm8k",
                )
            )
        return builders

    def _instruction_builders(self, index: int, count: int) -> list[EnvGroupBuilder]:
        builders: list[EnvGroupBuilder] = []
        start = index * max(count, 1)
        for offset in range(count):
            problem = self.if_problems[(start + offset) % len(self.if_problems)]
            builders.append(
                RepeatedEnvGroupBuilder(
                    env_thunk=partial(InstructionConstraintEnv, problem, self.renderer),
                    num_envs=self.group_size,
                    task_name="ifeval_synth",
                )
            )
        return builders

    def _code_builders(self, index: int, count: int) -> list[EnvGroupBuilder]:
        builders: list[EnvGroupBuilder] = []
        start = index * max(count, 1)
        for offset in range(count):
            problem = self.code_problems[(start + offset) % len(self.code_problems)]
            builders.append(
                RepeatedEnvGroupBuilder(
                    env_thunk=partial(CodeCheckEnv, problem, self.renderer),
                    num_envs=self.group_size,
                    task_name="code_synth",
                )
            )
        return builders


def compute_task_counts(
    *, groups_per_batch: int, math_weight: float, if_weight: float, code_weight: float
) -> dict[str, int]:
    weights = {"math": math_weight, "if": if_weight, "code": code_weight}
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Task weights must be non-negative")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("At least one task weight must be positive")

    raw = {task: groups_per_batch * weight / total_weight for task, weight in weights.items()}
    counts = {task: int(math.floor(value)) for task, value in raw.items()}
    remainder = groups_per_batch - sum(counts.values())
    for task, _ in sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True):
        if remainder <= 0:
            break
        counts[task] += 1
        remainder -= 1
    for task, weight in weights.items():
        if weight > 0 and counts[task] == 0:
            donor = max(counts, key=counts.get)
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[task] = 1
    return counts


def build_instruction_problems(seed: int) -> list[InstructionProblem]:
    topics = [
        "ocean conservation",
        "healthy study habits",
        "safe bicycle commuting",
        "community gardens",
        "planning a small team project",
        "reducing food waste",
        "preparing for a presentation",
        "organizing research notes",
    ]
    rng = random.Random(seed)
    rng.shuffle(topics)
    problems: list[InstructionProblem] = []
    for i, topic in enumerate(topics * 8):
        count = 2 + (i % 3)
        required = ["coral", "focus", "helmet", "soil", "timeline", "leftovers"][i % 6]
        forbidden = ["fish", "phone", "car", "plastic", "delay", "trash"][i % 6]
        problems.append(
            InstructionProblem(
                prompt=(
                    f"Write exactly {count} bullet points about {topic}. "
                    f"Every bullet must include the word '{required}'. "
                    f"Do not use the word '{forbidden}'. Use only bullet lines."
                ),
                kind="bullets_keyword",
                params={"count": count, "required": required, "forbidden": forbidden},
            )
        )
        keys = ("answer,reason" if i % 2 == 0 else "title,summary")
        problems.append(
            InstructionProblem(
                prompt=(
                    f"Respond about {topic} as valid JSON only. "
                    f"The object must contain exactly these keys: {keys}."
                ),
                kind="json_keys",
                params={"keys": keys},
            )
        )
        case = "upper" if i % 2 == 0 else "lower"
        keyword = ["steady", "clear", "brief", "careful"][i % 4]
        problems.append(
            InstructionProblem(
                prompt=(
                    f"Give one {case}case sentence about {topic}. "
                    f"Include the word '{keyword}' and use at most 12 words."
                ),
                kind="case_and_keyword",
                params={"case": case, "keyword": keyword, "max_words": 12},
            )
        )
    return problems


def build_code_problems() -> list[CodeProblem]:
    specs = [
        (
            "add_one",
            "Write a Python function `add_one(x)` that returns x plus one.",
            ("assert add_one(1) == 2", "assert add_one(-3) == -2", "assert add_one(0) == 1"),
        ),
        (
            "is_even",
            "Write a Python function `is_even(n)` that returns True if n is even and False otherwise.",
            ("assert is_even(2) is True", "assert is_even(7) is False", "assert is_even(0) is True"),
        ),
        (
            "first_char",
            "Write a Python function `first_char(text)` that returns the first character of a non-empty string.",
            ("assert first_char('abc') == 'a'", "assert first_char('Z') == 'Z'"),
        ),
        (
            "sum_list",
            "Write a Python function `sum_list(values)` that returns the sum of a list of numbers.",
            ("assert sum_list([1, 2, 3]) == 6", "assert sum_list([]) == 0", "assert sum_list([-1, 5]) == 4"),
        ),
        (
            "reverse_string",
            "Write a Python function `reverse_string(text)` that returns the string reversed.",
            ("assert reverse_string('abc') == 'cba'", "assert reverse_string('') == ''"),
        ),
        (
            "count_vowels",
            "Write a Python function `count_vowels(text)` that counts lowercase and uppercase vowels.",
            ("assert count_vowels('hello') == 2", "assert count_vowels('BCD') == 0", "assert count_vowels('AEi') == 3"),
        ),
        (
            "max_pair_sum",
            "Write a Python function `max_pair_sum(values)` that returns the sum of the two largest numbers.",
            ("assert max_pair_sum([1, 4, 2]) == 6", "assert max_pair_sum([-5, -1, -3]) == -4"),
        ),
        (
            "dedupe_preserve",
            "Write a Python function `dedupe_preserve(values)` that removes duplicates while preserving order.",
            ("assert dedupe_preserve([1, 2, 1, 3]) == [1, 2, 3]", "assert dedupe_preserve([]) == []"),
        ),
    ]
    return [
        CodeProblem(
            prompt=(
                spec
                + " Return only Python code. Do not include explanations or markdown fences."
            ),
            entry_point=name,
            tests=tests,
        )
        for name, spec, tests in specs
    ]


@chz.chz
class MultiTaskRLDatasetBuilder(RLDatasetBuilder):
    groups_per_batch: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    math_weight: float = 0.5
    if_weight: float = 0.3
    code_weight: float = 0.2
    seed: int = 0

    async def __call__(self) -> tuple[MultiTaskRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        dataset = MultiTaskRLDataset(
            groups_per_batch=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            math_weight=self.math_weight,
            if_weight=self.if_weight,
            code_weight=self.code_weight,
            seed=self.seed,
            split="train",
        )
        return dataset, None
