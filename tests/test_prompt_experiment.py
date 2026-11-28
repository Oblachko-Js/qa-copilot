from pathlib import Path
import unittest

from scripts.run_evaluation import load_dataset
from scripts.run_prompt_experiment import PRACTICES, build_experimental_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PromptExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        examples = load_dataset(PROJECT_ROOT / "evaluation/data/test_requirements.json")
        cls.example = next(example for example in examples if example.id == "QA-001")

    def test_baseline_preserves_the_original_application_prompt(self) -> None:
        expected_prompt = (
            "Составь краткий, но полный набор тест-кейсов для требования ниже. "
            "Все текстовые значения пиши на русском языке. "
            "Для priority используй только одно из значений: низкий, средний, высокий. "
            "Верни структурированные данные с полями title, preconditions, steps, "
            "expected_result и priority. "
            "Составь от 1 до 5 тест-кейсов.\n\n"
            f"Требование: {self.example.requirement}"
        )
        self.assertEqual(
            build_experimental_prompt(self.example, max_cases=5, step=0),
            expected_prompt,
        )

    def test_experiment_contains_all_nine_ordered_versions(self) -> None:
        self.assertEqual(len(PRACTICES), 9)
        prompts = [
            build_experimental_prompt(self.example, max_cases=5, step=step)
            for step in range(9)
        ]
        self.assertTrue(all(prompt for prompt in prompts))
        self.assertEqual(len(set(prompts)), 9)

    def test_each_version_adds_the_assigned_practice(self) -> None:
        expected_terms = {
            1: ("Цель:", "Задача:"),
            2: ("Инструкции:", "Не добавляй функции"),
            3: ("Роль:", "QA-инженер"),
            4: ("Контекст продукта:",),
            5: ("Входные данные:", "тип задачи:"),
            6: ("Формат каждого объекта test_cases:", "expected_result:"),
            7: ("<instructions>", "<input_data>", "</input_data>"),
            8: ("<decomposition>", "атомарные сценарии"),
        }
        for step, terms in expected_terms.items():
            with self.subTest(step=step):
                prompt = build_experimental_prompt(
                    self.example,
                    max_cases=5,
                    step=step,
                )
                for term in terms:
                    self.assertIn(term, prompt)

    def test_invalid_step_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_experimental_prompt(self.example, max_cases=5, step=9)


if __name__ == "__main__":
    unittest.main()
