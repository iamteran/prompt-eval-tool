"""
Unit tests for the Prompt Evaluator
"""

import unittest
import json
import os
from evaluator import PromptEvaluator

class TestPromptEvaluator(unittest.TestCase):
    
    def setUp(self):
        self.evaluator = PromptEvaluator()
    
    def test_evaluate_valid(self):
        result = self.evaluator.evaluate(
            prompt="Test prompt",
            response="Test response",
            scores={'helpfulness': 8, 'accuracy': 9, 'harmlessness': 10, 'clarity': 7}
        )
        self.assertEqual(result['scores']['helpfulness'], 8)
        self.assertEqual(result['total'], 8.5)
    
    def test_evaluate_invalid_score(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(
                prompt="Test",
                response="Test",
                scores={'helpfulness': 11, 'accuracy': 9, 'harmlessness': 10, 'clarity': 7}
            )
    
    def test_evaluate_missing_key(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(
                prompt="Test",
                response="Test",
                scores={'helpfulness': 8, 'accuracy': 9}
            )
    
    def test_clear(self):
        self.evaluator.evaluate(
            prompt="Test",
            response="Test",
            scores={'helpfulness': 8, 'accuracy': 9, 'harmlessness': 10, 'clarity': 7}
        )
        self.assertEqual(len(self.evaluator.results), 1)
        self.evaluator.clear()
        self.assertEqual(len(self.evaluator.results), 0)
    
    def test_get_average_scores(self):
        self.evaluator.evaluate(
            prompt="Test 1",
            response="Response 1",
            scores={'helpfulness': 8, 'accuracy': 9, 'harmlessness': 10, 'clarity': 7}
        )
        self.evaluator.evaluate(
            prompt="Test 2",
            response="Response 2",
            scores={'helpfulness': 6, 'accuracy': 7, 'harmlessness': 8, 'clarity': 6}
        )
        avg = self.evaluator.get_average_scores()
        self.assertEqual(avg['helpfulness'], 7.0)
    
    def test_search_by_prompt(self):
        self.evaluator.evaluate(
            prompt="What is Python?",
            response="Python is a programming language.",
            scores={'helpfulness': 8, 'accuracy': 9, 'harmlessness': 10, 'clarity': 7}
        )
        self.evaluator.evaluate(
            prompt="What is quantum computing?",
            response="Quantum computing uses qubits.",
            scores={'helpfulness': 8, 'accuracy': 9, 'harmlessness': 10, 'clarity': 7}
        )
        results = self.evaluator.search_by_prompt("Python")
        self.assertEqual(len(results), 1)

if __name__ == '__main__':
    unittest.main()
