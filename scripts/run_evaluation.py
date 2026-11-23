from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from app.main import TestCaseList


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "data" / "test_requirements.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "latest.csv"


class ReferenceScenario(BaseModel):
    id: str
    description: str
    expected_priority: str


class EvaluationExample(BaseModel):
    id: str
    complexity: str
    requirement: str = Field(min_length=10)
    reference_scenarios: list[ReferenceScenario] = Field(min_length=1)


def load_dataset(path: Path) -> list[EvaluationExample]:
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Не удалось прочитать датасет {path}: {exc}") from exc

    if not isinstance(raw_data, list) or not 10 <= len(raw_data) <= 20:
        raise ValueError("Датасет должен содержать от 10 до 20 требований.")

    examples = [EvaluationExample.model_validate(item) for item in raw_data]
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("В датасете обнаружены повторяющиеся идентификаторы.")

    return examples


def call_api(
    endpoint: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[int | None, str, float, str | None]:
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()

    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            return response.status, response_body, time.perf_counter() - started_at, None
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return exc.code, response_body, time.perf_counter() - started_at, None
    except URLError as exc:
        error = f"Не удалось подключиться к API: {exc.reason}"
        return None, "", time.perf_counter() - started_at, error
    except TimeoutError:
        error = f"Истекло время ожидания ответа ({timeout} сек.)"
        return None, "", time.perf_counter() - started_at, error


def evaluate_example(
    example: EvaluationExample,
    endpoint: str,
    model: str,
    temperature: float,
    max_cases: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "requirement": example.requirement,
        "temperature": temperature,
        "max_cases": max_cases,
    }
    status, response_body, elapsed, error = call_api(endpoint, payload, timeout)
    parsed_response: dict[str, Any] | None = None
    schema_valid = False
    case_count = 0

    if status == 200:
        try:
            parsed = TestCaseList.model_validate_json(response_body)
            parsed_response = parsed.model_dump(mode="json")
            case_count = len(parsed.test_cases)
            schema_valid = case_count <= max_cases
            if not schema_valid:
                error = f"API вернул {case_count} кейсов при ограничении {max_cases}."
        except ValidationError as exc:
            error = f"Ответ не соответствует Pydantic-схеме: {exc}"

    return {
        "id": example.id,
        "complexity": example.complexity,
        "model": model,
        "endpoint": endpoint,
        "requirement": example.requirement,
        "reference_scenarios_json": json.dumps(
            [scenario.model_dump() for scenario in example.reference_scenarios],
            ensure_ascii=False,
        ),
        "request_json": json.dumps(payload, ensure_ascii=False),
        "response_status": status,
        "latency_seconds": round(elapsed, 3),
        "schema_valid": schema_valid,
        "test_case_count": case_count,
        "response_json": (
            json.dumps(parsed_response, ensure_ascii=False)
            if parsed_response is not None
            else response_body
        ),
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогон синтетического датасета через API qa-copilot."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--endpoint",
        default=os.getenv(
            "QA_COPILOT_API_URL",
            "http://127.0.0.1:8000/generate-testcases",
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "не указан"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.temperature <= 2:
        raise ValueError("Температура должна быть в диапазоне от 0 до 2.")
    if not 1 <= args.max_cases <= 15:
        raise ValueError("Максимальное число кейсов должно быть от 1 до 15.")
    if args.timeout <= 0:
        raise ValueError("Таймаут должен быть положительным числом.")

    examples = load_dataset(args.dataset)
    rows = []

    for index, example in enumerate(examples, start=1):
        row = evaluate_example(
            example=example,
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            max_cases=args.max_cases,
            timeout=args.timeout,
        )
        rows.append(row)
        outcome = (
            "OK"
            if row["response_status"] == 200 and row["schema_valid"]
            else "ОШИБКА"
        )
        print(
            f"[{index}/{len(examples)}] {example.id}: {outcome}; "
            f"HTTP {row['response_status']}; {row['latency_seconds']:.1f} сек."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")

    passed = sum(row["schema_valid"] for row in rows)
    print(f"\nРезультат сохранён: {args.output}")
    print(f"Успешно прошли проверку схемы: {passed}/{len(rows)}")


if __name__ == "__main__":
    main()
