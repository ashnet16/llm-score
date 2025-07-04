def self_reflection_prompt_template(question: str, answer: str) -> str:
    return f"""
You are evaluating the accuracy of a language model's answer.

Question: {question}

Answer: {answer}

As a reflection model, you must consider whether the answer contains factual information that can be verified and whether it reasonably addresses the question. 

Importantly, if the question is subjective, ambiguous, or lacks enough context (such as user-specific knowledge), you should err on the side of caution.

How likely is the answer to be correct?

Respond with one of the following options only: Very likely, Somewhat likely, Unlikely.
"""



def answer_prompt_template(question: str) -> str:
    return f"""
    You are a language model tasked with answering questions as accurately as possible.

    Please answer the following question:

    {question}

    """
