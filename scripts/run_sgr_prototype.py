from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.main import TestCaseList, get_llm_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "sgr-example.json"
DEFAULT_REQUIREMENT = (
    "Если сумма товаров составляет 5000 рублей или больше, доставка бесплатна. "
    "Если сумма товаров меньше 5000 рублей, стоимость доставки составляет 300 рублей."
)


class ScenarioCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        min_length=1,
        description="Проверяемый сценарий, следующий из требования.",
    )
    scenario_type: Literal["positive", "negative", "boundary"] = Field(
        description="Тип сценария: позитивный, негативный или граничный.",
    )
    risk: str = Field(
        min_length=1,
        description="Краткое объяснение риска, который проверяет сценарий.",
    )


class RequirementAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_entities: list[str] = Field(
        min_length=1,
        description="Сущности и участники, прямо названные в требовании.",
    )
    constraints: list[str] = Field(
        description="Явные числовые и бизнес-ограничения.",
    )
    ambiguities: list[str] = Field(
        description="Недостающие детали или неоднозначности требования; может быть пустым.",
    )
    scenarios: list[ScenarioCandidate] = Field(
        min_length=1,
        max_length=15,
        description="Кандидаты в тестовые сценарии, выведенные из требования.",
    )


_ModelT = TypeVar("_ModelT", bound=BaseModel)

ANALYSIS_SYSTEM_PROMPT = """Ты анализируешь требование для QA-инженера.
Извлеки только факты, явно содержащиеся в требовании.
Перечисли ключевые сущности, ограничения и неоднозначности.
Считай неоднозначностью только отсутствующую информацию, мешающую задать ожидаемый результат.
Не записывай как неоднозначность прямые следствия условия; например, «5000 рублей или больше» включает значения выше 5000.
Сформируй атомарные позитивные, негативные и граничные сценарии, когда они применимы.
Для числовых порогов рассмотри точное граничное значение и применимые значения ниже и выше него.
Не включай дублирующие сценарии.
Для каждого сценария кратко укажи риск, который он проверяет.
Не превращай неоднозначность в придуманное правило и не создавай тест-кейсы на этом шаге.
Верни только данные, соответствующие схеме RequirementAnalysis."""

GENERATION_SYSTEM_PROMPT = """Ты опытный QA-инженер.
Сформируй от 1 до {max_cases} тест-кейсов на русском языке.
Исходное требование — главный источник истины; используй анализ как список кандидатов в сценарии.
Если сценарий или анализ противоречит явно заданному правилу требования, исправь сценарий и следуй требованию.
Сформируй ровно один тест-кейс на каждый выбранный сценарий из анализа, не объединяй и не дублируй сценарии.
Если сценариев больше лимита, выбери наиболее важные риски и создай по одному кейсу для каждого выбранного сценария.
Не добавляй поведение, правила и ограничения, которых нет в исходном требовании.
Для каждого кейса верни title, preconditions, steps, expected_result и priority.
steps должен быть непустым списком, priority — одним из значений: низкий, средний, высокий.
Верни только данные, соответствующие схеме TestCaseList."""


def _extract_parsed(
    completion: Any,
    response_model: type[_ModelT],
    stage: str,
) -> _ModelT:
    if not completion.choices:
        raise ValueError(f"Модель вернула пустой ответ на этапе «{stage}».")

    parsed = completion.choices[0].message.parsed
    if not isinstance(parsed, response_model):
        raise ValueError(
            f"Ответ модели не прошёл проверку схемы на этапе «{stage}»."
        )
    return parsed


def analyze_requirement(
    client: OpenAI,
    requirement: str,
    model: str,
    temperature: float,
) -> RequirementAnalysis:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<requirement>\n{requirement}\n</requirement>",
            },
        ],
        temperature=temperature,
        response_format=RequirementAnalysis,
    )
    return _extract_parsed(completion, RequirementAnalysis, "анализ требования")


def generate_test_cases(
    client: OpenAI,
    requirement: str,
    analysis: RequirementAnalysis,
    model: str,
    temperature: float,
    max_cases: int,
) -> TestCaseList:
    analysis_json = analysis.model_dump_json(ensure_ascii=False, indent=2)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": GENERATION_SYSTEM_PROMPT.format(max_cases=max_cases),
            },
            {
                "role": "user",
                "content": (
                    "<generation_input>\n"
                    f"<requirement>\n{requirement}\n</requirement>\n"
                    "<validated_analysis>\n"
                    f"{analysis_json}\n"
                    "</validated_analysis>\n"
                    "</generation_input>"
                ),
            },
        ],
        temperature=temperature,
        response_format=TestCaseList,
    )
    test_cases = _extract_parsed(completion, TestCaseList, "генерация тест-кейсов")
    expected_count = min(len(analysis.scenarios), max_cases)
    if len(test_cases.test_cases) != expected_count:
        raise ValueError(
            f"Модель вернула {len(test_cases.test_cases)} кейсов, "
            f"ожидалось {expected_count} при ограничении {max_cases}."
        )
    return test_cases


def run_sgr(
    client: OpenAI,
    requirement: str,
    model: str,
    temperature: float,
    max_cases: int,
) -> tuple[RequirementAnalysis, TestCaseList]:
    analysis = analyze_requirement(client, requirement, model, temperature)
    test_cases = generate_test_cases(
        client=client,
        requirement=requirement,
        analysis=analysis,
        model=model,
        temperature=temperature,
        max_cases=max_cases,
    )
    return analysis, test_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Двухшаговый SGR-прототип: анализ требования и генерация тест-кейсов."
    )
    parser.add_argument("--requirement", default=DEFAULT_REQUIREMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.temperature <= 2:
        raise ValueError("Температура должна быть в диапазоне от 0 до 2.")
    if not 1 <= args.max_cases <= 15:
        raise ValueError("Максимум кейсов должен быть от 1 до 15.")

    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("Не задана переменная окружения LLM_MODEL.")
    client = get_llm_client()
    analysis, test_cases = run_sgr(
        client=client,
        requirement=args.requirement,
        model=model,
        temperature=args.temperature,
        max_cases=args.max_cases,
    )

    result = {
        "model": model,
        "temperature": args.temperature,
        "max_cases": args.max_cases,
        "requirement": args.requirement,
        "analysis": analysis.model_dump(mode="json"),
        "test_cases": test_cases.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nРезультат сохранён: {args.output}")


if __name__ == "__main__":
    main()
