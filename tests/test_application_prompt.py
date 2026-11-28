import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from app.main import (
    SYSTEM_PROMPT,
    GenerateRequest,
    TestCase,
    TestCaseList,
    build_prompt,
    generate_testcases,
)
from scripts.run_evaluation import load_dataset
from scripts.run_prompt_experiment import build_experimental_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ApplicationPromptTests(unittest.TestCase):
    def test_system_prompt_contains_all_eight_practices(self) -> None:
        for expected in (
            "<instructions>",
            "<product_context>",
            "<decomposition>",
            "Цель:",
            "Задача:",
            "Инструкции:",
            "Роль:",
            "Формат каждого объекта test_cases:",
            "Выдели атомарные сценарии",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, SYSTEM_PROMPT)

    def test_user_prompt_contains_structured_dynamic_input(self) -> None:
        prompt = build_prompt("Пользователь может сбросить пароль.", max_cases=3)

        self.assertIn("<input_data>", prompt)
        self.assertIn("- максимальное число кейсов: 3;", prompt)
        self.assertIn("- текст требования: Пользователь может сбросить пароль.", prompt)
        self.assertIn("</input_data>", prompt)

    def test_application_prompt_matches_final_experiment_version(self) -> None:
        example = load_dataset(
            PROJECT_ROOT / "evaluation/data/test_requirements.json"
        )[0]
        experimental_prompt = build_experimental_prompt(
            example,
            max_cases=5,
            step=8,
        )
        application_blocks = {
            "instructions": SYSTEM_PROMPT.format(max_cases=5),
            "product_context": SYSTEM_PROMPT.format(max_cases=5),
            "input_data": build_prompt(example.requirement, max_cases=5),
            "decomposition": SYSTEM_PROMPT.format(max_cases=5),
        }

        for tag, application_prompt in application_blocks.items():
            with self.subTest(tag=tag):
                self.assertEqual(
                    self._extract_block(application_prompt, tag),
                    self._extract_block(experimental_prompt, tag),
                )

    @staticmethod
    def _extract_block(prompt: str, tag: str) -> str:
        opening_tag = f"<{tag}>"
        closing_tag = f"</{tag}>"
        start = prompt.index(opening_tag)
        end = prompt.index(closing_tag, start) + len(closing_tag)
        return prompt[start:end]

    def test_endpoint_sends_system_prompt_and_schema_constrained_user_input(self) -> None:
        generated_case = TestCase(
            title="Сброс пароля",
            preconditions="Пользователь зарегистрирован.",
            steps=["Запросить сброс пароля."],
            expected_result="Пользователь получает ссылку.",
            priority="средний",
        )
        client = Mock()
        client.beta.chat.completions.parse.return_value.choices = [
            Mock(message=Mock(parsed=TestCaseList(test_cases=[generated_case])))
        ]
        payload = GenerateRequest(
            requirement="Пользователь может сбросить пароль.",
            temperature=0.2,
            max_cases=3,
        )

        with (
            patch("app.main.get_llm_client", return_value=client),
            patch.dict(os.environ, {"LLM_MODEL": "test-model"}),
        ):
            response = generate_testcases(payload)

        self.assertEqual(response.test_cases, [generated_case])
        call = client.beta.chat.completions.parse.call_args
        self.assertEqual(
            call.kwargs["messages"],
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(max_cases=3),
                },
                {
                    "role": "user",
                    "content": build_prompt(payload.requirement, max_cases=3),
                },
            ],
        )
        self.assertIs(call.kwargs["response_format"], TestCaseList)


if __name__ == "__main__":
    unittest.main()
