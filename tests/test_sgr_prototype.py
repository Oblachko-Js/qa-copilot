import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from app.main import TestCase, TestCaseList
from scripts.run_sgr_prototype import (
    RequirementAnalysis,
    ScenarioCandidate,
    run_sgr,
)


def _completion(parsed: object) -> Mock:
    return Mock(choices=[Mock(message=Mock(parsed=parsed))])


class SgrPrototypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = RequirementAnalysis(
            key_entities=["заказ", "бонусный балл"],
            constraints=["1 балл за каждые полные 100 рублей"],
            ambiguities=["Не указано, сколько рублей соответствует одному баллу."],
            scenarios=[
                ScenarioCandidate(
                    description="Проверить начисление балла после завершения заказа.",
                    scenario_type="positive",
                    risk="Проверяется корректное начисление.",
                )
            ],
        )
        self.test_case = TestCase(
            title="Начисление бонусных баллов",
            preconditions="Оплаченный заказ на 100 рублей завершён.",
            steps=["Проверить баланс бонусных баллов."],
            expected_result="Начислен один бонусный балл.",
            priority="высокий",
        )

    def test_sgr_makes_two_schema_constrained_calls_in_order(self) -> None:
        client = Mock()
        client.beta.chat.completions.parse.side_effect = [
            _completion(self.analysis),
            _completion(TestCaseList(test_cases=[self.test_case])),
        ]

        analysis, test_cases = run_sgr(
            client=client,
            requirement="За полные 100 рублей начисляется один бонусный балл.",
            model="test-model",
            temperature=0.2,
            max_cases=3,
        )

        self.assertEqual(analysis, self.analysis)
        self.assertEqual(test_cases.test_cases, [self.test_case])
        calls = client.beta.chat.completions.parse.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0].kwargs["response_format"], RequirementAnalysis)
        self.assertIs(calls[1].kwargs["response_format"], TestCaseList)
        self.assertIn(
            "бонусный балл",
            calls[1].kwargs["messages"][1]["content"],
        )
        self.assertIn(
            '"scenario_type": "positive"',
            calls[1].kwargs["messages"][1]["content"],
        )

    def test_invalid_scenario_type_is_rejected_by_pydantic(self) -> None:
        with self.assertRaises(ValidationError):
            ScenarioCandidate(
                description="Неизвестный тип сценария.",
                scenario_type="exploratory",
                risk="Проверка типов.",
            )

    def test_model_exceeding_requested_case_limit_fails_explicitly(self) -> None:
        client = Mock()
        client.beta.chat.completions.parse.side_effect = [
            _completion(self.analysis),
            _completion(
                TestCaseList(test_cases=[self.test_case, self.test_case])
            ),
        ]

        with self.assertRaisesRegex(ValueError, "при ограничении 1"):
            run_sgr(
                client=client,
                requirement="За полные 100 рублей начисляется один бонусный балл.",
                model="test-model",
                temperature=0.2,
                max_cases=1,
            )


if __name__ == "__main__":
    unittest.main()
