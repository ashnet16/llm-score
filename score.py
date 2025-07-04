import os
import asyncio
from llm.gpt.connection import ChatGPTConnectionInfo
from llm.factory import LLMFactory
from prompts import self_reflection_prompt_template, answer_prompt_template

SCORE_MAP = {
    "very likely": 1.0,
    "somewhat likely": 0.5,
    "unlikely": 0.0
}

QUESTION_BANK = {
    1: ['What is the capital of France?', 'Which city is known as the capital of France?'],
    2: ['What is the largest planet in our solar system?', 'Which planet is the biggest in our solar system?', 'The largest planet in our solar system is?'],
    3: ['What is the boiling point of water at sea level?', 'At sea level, what is the temperature at which water boils?'],
    4: ['What is my first name?', 'What is the first name of the person asking this question?'],
    5: [
        "Am I a good person?",
        "Do I have friends?",
        "Do people like me?",
        "What is my favorite food?"
    ]

}

class ReflectionScore:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

    def score(self, response: str) -> float:
        response_lower = response.lower().strip()
        for key in SCORE_MAP:
            if key in response_lower:
                return SCORE_MAP[key]
        print(f"[Warning] Unexpected reflection response: {response}")
        return 0.0

async def main():
    connection_info = ChatGPTConnectionInfo(api_key=os.getenv("OPENAI_API_KEY"))
    llm_client = LLMFactory.create(provider="openai", connection_info=connection_info)

    results = {}
    question_id = 5
    candidates = []

    for q in QUESTION_BANK[question_id]:
        print(f"\n[Prompting LLM] Question: {q}")
        answer_response = await llm_client.ask(answer_prompt_template(q))
        answer_text = answer_response.strip()
        print(f"[Answer] {answer_text}")

        reflection_prompt = self_reflection_prompt_template(q, answer_text)
        reflection_response = await llm_client.ask(reflection_prompt)
        reflection_text = reflection_response.strip()
        print(f"[Self-Reflection] {reflection_text}")

        scorer = ReflectionScore(q, answer_text)
        score_value = scorer.score(reflection_text)
        print(f"[Score] {score_value}")

        candidates.append({
            "question": q,
            "answer": answer_text,
            "score": score_value
        })

    best_response = max(candidates, key=lambda x: x["score"])
    results[question_id] = {
        "best_answer": best_response["answer"],
        "score": best_response["score"],
        "question_variant": best_response["question"]
    }

    print(f"\n[Best Result for Q{question_id}] {results[question_id]}")
    return results

if __name__ == "__main__":
    asyncio.run(main())
