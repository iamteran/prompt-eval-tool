"""
Example: Multi-turn conversation testing with the Prompt Evaluator
"""

from evaluator import PromptEvaluator

def run_multi_turn_test():
    """Simulate a multi-turn conversation test."""
    evaluator = PromptEvaluator()
    
    print("=== MULTI-TURN CONVERSATION TEST ===")
    print("Testing a travel assistant scenario...\n")
    
    # Turn 1
    evaluator.evaluate(
        prompt="[Turn 1] I'm planning a trip to Paris next month. Can you help?",
        response="Of course! What would you like to know about?",
        scores={'helpfulness': 9, 'accuracy': 10, 'harmlessness': 10, 'clarity': 9}
    )
    print("✓ Turn 1: Good initial response")
    
    # Turn 2 — contains factual error
    evaluator.evaluate(
        prompt="[Turn 2] What are the must-see attractions?",
        response="The Eiffel Tower, Louvre, and Notre-Dame. The Eiffel Tower was completed in 1887.",
        scores={'helpfulness': 7, 'accuracy': 6, 'harmlessness': 10, 'clarity': 8}
    )
    print("⚠ Turn 2: Factual error detected (Eiffel Tower completion year)")
    
    # Turn 3 — context loss
    evaluator.evaluate(
        prompt="[Turn 3] Actually, I'm more interested in food experiences.",
        response="Definitely! Also, visit the Eiffel Tower at sunset.",
        scores={'helpfulness': 6, 'accuracy': 8, 'harmlessness': 10, 'clarity': 7}
    )
    print("⚠ Turn 3: Context loss — user shifted to food, model still recommended Eiffel Tower")
    
    # Turn 4 — good recovery
    evaluator.evaluate(
        prompt="[Turn 4] Any food markets or cooking classes?",
        response="Check out Rue Mouffetard market! La Cuisine Paris offers great classes.",
        scores={'helpfulness': 9, 'accuracy': 9, 'harmlessness': 10, 'clarity': 9}
    )
    print("✓ Turn 4: Good recovery and relevant recommendations")
    
    print("\n" + "="*50)
    summary = evaluator.summary()
    print(f"Total turns: {summary['count']}")
    print(f"Average helpfulness: {summary['avg_scores']['helpfulness']:.1f}/10")
    print(f"Average accuracy: {summary['avg_scores']['accuracy']:.1f}/10")
    
    evaluator.export('multi_turn_results.json')
    print("\nResults exported to multi_turn_results.json")

if __name__ == "__main__":
    run_multi_turn_test()
