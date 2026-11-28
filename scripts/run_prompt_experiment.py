from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from app.main import TestCaseList
from scripts.run_evaluation import EvaluationExample, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "prompt-experiment-results.csv"
CSV_FIELDS = [
    "step",
    "practice",
    "case_id",
    "complexity",
    "model",
    "temperature",
    "max_cases",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_seconds",
    "schema_valid",
    "prompt",
    "requirement",
    "response_json",
    "error",
]


PRACTICES = [
    "Исходный промпт",
    "+ явная цель и задача",
    "+ конкретные инструкции",
    "+ роль QA-инженера",
    "+ контекст продукта",
    "+ структурированные входные данные",
    "+ описание формата выхода",
    "+ разделители блоков",
    "+ декомпозиция на сценарии",
]


def build_experimental_prompt(
    example: EvaluationExample,
    max_cases: int,
    step: int,
) -> str:
    if not 0 <= step <= 8:
        raise ValueError("Шаг эксперимента должен быть от 0 до 8.")

    if step == 0:
        return (
            "Составь краткий, но полный набор тест-кейсов для требования ниже. "
            "Все текстовые значения пиши на русском языке. "
            "Для priority используй только одно из значений: низкий, средний, высокий. "
            "Верни структурированные данные с полями title, preconditions, steps, "
            "expected_result и priority. "
            f"Составь от 1 до {max_cases} тест-кейсов.\n\n"
            f"Требование: {example.requirement}"
        )

    prompt = [
        "Составь краткий, но полный набор тест-кейсов для требования ниже.",
        "Все текстовые значения пиши на русском языке.",
        "Для priority используй только одно из значений: низкий, средний, высокий.",
        "Верни структурированные данные с полями title, preconditions, steps, "
        "expected_result и priority.",
        f"Составь от 1 до {max_cases} тест-кейсов.",
    ]

    if step >= 1:
        prompt.extend(
            [
                "Цель: помочь QA-инженеру подготовить проверки значимых рисков "
                "из требования, не добавляя неподтверждённое поведение.",
                "Задача: выделить тестовые сценарии и превратить их в проверяемые "
                "тест-кейсы.",
            ]
        )

    if step >= 2:
        prompt.extend(
            [
                "Инструкции:",
                "- Каждый кейс должен проверять отдельное поведение.",
                "- Для применимых условий проверь позитивный, негативный и граничный сценарии.",
                "- Не добавляй функции, правила и ограничения, которых нет во входных данных.",
                "- Шаги формулируй как конкретные действия; ожидаемый результат — как наблюдаемое поведение.",
                "- Не повторяй один сценарий другими словами.",
            ]
        )

    if step >= 3:
        prompt.append(
            "Роль: ты опытный QA-инженер, который анализирует требования "
            "и составляет точные, воспроизводимые проверки."
        )

    if step >= 4:
        prompt.extend(
            [
                "Контекст продукта: QA Copilot помогает тестировать программные продукты "
                "и преобразует описание функции в тест-кейсы.",
                "Известно только то, что явно сообщено во входном требовании; "
                "не предполагай наличие неописанных экранов, ролей или правил.",
            ]
        )

    if step >= 5:
        requirement_block = (
            "Входные данные:\n"
            "- тип задачи: генерация тест-кейсов по требованию;\n"
            f"- максимальное число кейсов: {max_cases};\n"
            f"- текст требования: {example.requirement}"
        )
    else:
        requirement_block = f"Требование: {example.requirement}"

    if step >= 6:
        prompt.extend(
            [
                "Формат каждого объекта test_cases:",
                "- title: краткое название сценария;",
                "- preconditions: условия до начала проверки;",
                "- steps: непустой список конкретных действий;",
                "- expected_result: наблюдаемый результат;",
                "- priority: ровно одно значение из списка «низкий», «средний», «высокий».",
                "Верни только данные, соответствующие заданной JSON-схеме.",
            ]
        )

    if step >= 7:
        instruction_lines = [
            line
            for line in prompt
            if not line.startswith(("Контекст продукта:", "Известно только то"))
        ]
        prompt = [
            "<instructions>\n"
            + "\n".join(instruction_lines)
            + "\n</instructions>",
            "<product_context>\n"
            + (
                "QA Copilot помогает тестировать программные продукты "
                "и преобразует описание функции в тест-кейсы. "
                "Известно только то, что явно сообщено во входном требовании; "
                "не предполагай наличие неописанных экранов, ролей или правил."
                if step >= 4
                else "Продуктовый контекст не задан."
            )
            + "\n</product_context>",
            "<input_data>\n" + requirement_block + "\n</input_data>",
        ]

    if step >= 8:
        prompt.append(
            "<decomposition>\n"
            "Перед формированием ответа последовательно:\n"
            "1. Выдели атомарные сценарии, прямо следующие из требования.\n"
            "2. Проверь, какие из них позитивные, негативные и граничные, если такие применимы.\n"
            "3. Для каждого выбранного сценария сформируй один отдельный тест-кейс.\n"
            "4. Удали повторы и сценарии, основанные на неподтверждённых предположениях.\n"
            "В ответ включи только итоговый список тест-кейсов, не промежуточный анализ.\n"
            "</decomposition>"
        )

    if step >= 7:
        return "\n\n".join(prompt)

    return "\n\n".join([*prompt, requirement_block])


def load_completed_rows(path: Path, model: str) -> set[tuple[int, str]]:
    if not path.exists():
        return set()

    with path.open(newline="", encoding="utf-8-sig") as file:
        return {
            (int(row["step"]), row["case_id"])
            for row in csv.DictReader(file)
            if row.get("model") == model and row.get("schema_valid") == "True"
        }


def append_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(result)
        file.flush()
        os.fsync(file.fileno())


def run_one(
    client: OpenAI,
    example: EvaluationExample,
    step: int,
    model: str,
    temperature: float,
    max_cases: int,
) -> dict[str, Any]:
    prompt = build_experimental_prompt(example, max_cases, step)
    started_at = time.perf_counter()
    result: dict[str, Any] = {
        "step": step,
        "practice": PRACTICES[step],
        "case_id": example.id,
        "complexity": example.complexity,
        "model": model,
        "temperature": temperature,
        "max_cases": max_cases,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "latency_seconds": None,
        "schema_valid": False,
        "prompt": prompt,
        "requirement": example.requirement,
        "response_json": "",
        "error": "",
    }

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format=TestCaseList,
        )
        result["latency_seconds"] = round(time.perf_counter() - started_at, 3)
        if completion.usage is not None:
            result["prompt_tokens"] = completion.usage.prompt_tokens
            result["completion_tokens"] = completion.usage.completion_tokens
            result["total_tokens"] = completion.usage.total_tokens

        if not completion.choices:
            raise ValueError("Модель вернула пустой список вариантов ответа.")

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Ответ модели не прошёл разбор по Pydantic-схеме.")
        if len(parsed.test_cases) > max_cases:
            raise ValueError(
                f"Модель вернула {len(parsed.test_cases)} кейсов при ограничении {max_cases}."
            )

        result["schema_valid"] = True
        result["response_json"] = json.dumps(
            parsed.model_dump(mode="json"),
            ensure_ascii=False,
        )
    except (OpenAIError, ValidationError, ValueError, TypeError) as exc:
        result["latency_seconds"] = round(time.perf_counter() - started_at, 3)
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнить последовательные версии промпта на фиксированном датасете."
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation/data/test_requirements.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen2.5:7b"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "ollama"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.temperature <= 2:
        raise ValueError("Температура должна быть в диапазоне от 0 до 2.")
    if not 1 <= args.max_cases <= 15:
        raise ValueError("Максимум кейсов должен быть в диапазоне от 1 до 15.")
    if args.timeout <= 0:
        raise ValueError("Таймаут должен быть положительным числом.")

    examples = load_dataset(args.dataset)
    if len(examples) != 12:
        raise ValueError(
            "Для сопоставимости эксперимент ожидает фиксированный набор из 12 требований."
        )

    client = OpenAI(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,
    )
    completed = load_completed_rows(args.output, args.model)
    total = len(PRACTICES) * len(examples)
    done = 0

    for step, practice in enumerate(PRACTICES):
        print(f"\nШаг {step}: {practice}", flush=True)
        for example in examples:
            if (step, example.id) in completed:
                done += 1
                print(
                    f"[{done}/{total}] {example.id}: уже выполнено",
                    flush=True,
                )
                continue

            result = run_one(
                client=client,
                example=example,
                step=step,
                model=args.model,
                temperature=args.temperature,
                max_cases=args.max_cases,
            )
            append_result(args.output, result)
            done += 1

            status = "OK" if result["schema_valid"] else "ОШИБКА"
            total_tokens = result["total_tokens"]
            token_text = str(total_tokens) if total_tokens is not None else "нет usage"
            print(
                f"[{done}/{total}] {example.id}: {status}; "
                f"{result['latency_seconds']:.1f} с; {token_text} токенов",
                flush=True,
            )

    print(f"\nЭксперимент завершён. Результаты: {args.output}", flush=True)


if __name__ == "__main__":
    main()
