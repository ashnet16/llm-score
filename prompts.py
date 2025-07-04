def self_reflection_prompt_template(question: str, answer: str) -> str:
    
   return f"""
    You are evaluating the correctness of an answer provided by a language model.

    Question: {question}

    Answer: {answer}

    How likely is the answer to be correct?

    Respond with one of the following options only: Very likely, Somewhat likely, Unlikely.
    """



def answer_prompt_template(question: str) -> str:
    return f"""
    You are a language model tasked with answering questions as accurately as possible.

    Please answer the following question:

    {question}

    """
