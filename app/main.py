from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, title="Название")
    preconditions: str = Field(..., min_length=1, title="Предусловия")
    steps: list[str] = Field(..., min_length=1, title="Шаги")
    expected_result: str = Field(..., min_length=1, title="Ожидаемый результат")
    priority: Literal["низкий", "средний", "высокий"] = Field(
        "средний",
        title="Приоритет",
    )


class TestCaseList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_cases: list[TestCase] = Field(
        ...,
        min_length=1,
        max_length=15,
        title="Список тест-кейсов",
    )


class GenerateRequest(BaseModel):
    requirement: str = Field(
        ...,
        min_length=10,
        title="Требование",
        description="Текст требования, по которому нужно составить тест-кейсы.",
    )
    temperature: float = Field(
        0.2,
        ge=0.0,
        le=2.0,
        title="Температура",
        description="Степень вариативности ответа модели: от 0 до 2.",
    )
    max_cases: int = Field(
        5,
        ge=1,
        le=15,
        title="Количество тест-кейсов",
        description="Максимальное число тест-кейсов в ответе: от 1 до 15.",
    )


app = FastAPI(
    title="QA Copilot — генерация тест-кейсов",
    description="API для генерации тест-кейсов на русском языке по тексту требования.",
    version="0.1.0",
)


def get_llm_client() -> OpenAI:
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")

    if not base_url or not api_key or not model:
        raise RuntimeError(
            "Не заданы настройки LLM. Укажите LLM_BASE_URL, LLM_API_KEY и LLM_MODEL."
        )

    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)


SYSTEM_PROMPT = """<instructions>
Составь краткий, но полный набор тест-кейсов для требования ниже.
Все текстовые значения пиши на русском языке.
Для priority используй только одно из значений: низкий, средний, высокий.
Верни структурированные данные с полями title, preconditions, steps, expected_result и priority.
Составь от 1 до {max_cases} тест-кейсов.
Цель: помочь QA-инженеру подготовить проверки значимых рисков из требования, не добавляя неподтверждённое поведение.
Задача: выделить тестовые сценарии и превратить их в проверяемые тест-кейсы.
Инструкции:
- Каждый кейс должен проверять отдельное поведение.
- Для применимых условий проверь позитивный, негативный и граничный сценарии.
- Не добавляй функции, правила и ограничения, которых нет во входных данных.
- Шаги формулируй как конкретные действия; ожидаемый результат — как наблюдаемое поведение.
- Не повторяй один сценарий другими словами.
Роль: ты опытный QA-инженер, который анализирует требования и составляет точные, воспроизводимые проверки.
Формат каждого объекта test_cases:
- title: краткое название сценария;
- preconditions: условия до начала проверки;
- steps: непустой список конкретных действий;
- expected_result: наблюдаемый результат;
- priority: ровно одно значение из списка «низкий», «средний», «высокий».
Верни только данные, соответствующие заданной JSON-схеме.
</instructions>

<product_context>
QA Copilot помогает тестировать программные продукты и преобразует описание функции в тест-кейсы. Известно только то, что явно сообщено во входном требовании; не предполагай наличие неописанных экранов, ролей или правил.
</product_context>

<decomposition>
Перед формированием ответа последовательно:
1. Выдели атомарные сценарии, прямо следующие из требования.
2. Проверь, какие из них позитивные, негативные и граничные, если такие применимы.
3. Для каждого выбранного сценария сформируй один отдельный тест-кейс.
4. Удали повторы и сценарии, основанные на неподтверждённых предположениях.
В ответ включи только итоговый список тест-кейсов, не промежуточный анализ.
</decomposition>"""


def build_prompt(requirement: str, max_cases: int) -> str:
    return (
        "<input_data>\n"
        "Входные данные:\n"
        "- тип задачи: генерация тест-кейсов по требованию;\n"
        f"- максимальное число кейсов: {max_cases};\n"
        f"- текст требования: {requirement}\n"
        "</input_data>"
    )


@app.get("/health", summary="Проверить работу сервиса")
def health() -> dict[str, str]:
    return {"status": "работает"}


@app.post(
    "/generate-testcases",
    summary="Сгенерировать тест-кейсы",
    description="Создаёт тест-кейсы на русском языке по тексту требования.",
    response_model=TestCaseList,
)
def generate_testcases(payload: GenerateRequest) -> TestCaseList:
    try:
        client = get_llm_client()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Сервис языковой модели недоступен: {exc}",
        ) from exc

    try:
        completion = client.beta.chat.completions.parse(
            model=os.environ["LLM_MODEL"],
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(max_cases=payload.max_cases),
                },
                {
                    "role": "user",
                    "content": build_prompt(payload.requirement, payload.max_cases),
                },
            ],
            temperature=payload.temperature,
            response_format=TestCaseList,
        )

        if not completion.choices:
            raise ValueError("Модель вернула пустой ответ")

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Не удалось разобрать ответ модели по заданной схеме")

        return TestCaseList(test_cases=parsed.test_cases[: payload.max_cases])
    except (OpenAIError, ValidationError, ValueError, TypeError, IndexError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось сгенерировать тест-кейсы: {exc}",
        ) from exc
