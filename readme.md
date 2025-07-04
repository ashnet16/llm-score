# Confidence Scoring for Large Language Models (LLMs)

## Overview
Based on the paper [Confidence Scoring for Large Language Models](https://arxiv.org/abs/2309.17066) by [Yin et al.](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+W.) (2023).

## Problem

A blocker to using LLMs in high-stakes applications is the **uncertainty of responses**, often due to:

- Hallucinations, which may stem from:
  - Vague prompts
  - Lack of training data
- Overconfidence in incorrect answers

## Challenge

LLM APIs like GPT are **black boxes**:

- No access to token-level probabilities  
- No access to training data  
- Cannot retrain or fine-tune the model

## Solution: Estimate Confidence Score

Use a confidence scoring function: C(x, y) → confidence score

Where:
- `x`: the input/prompt  
- `y`: the LLM's answer  
- `C`: derived from two components:
  - Observed Consistency (OC)
  - Self-Reflection Certainty (SRC)


## 1. Observed Consistency (OC) — Extrinsic Evaluation

- Generate multiple answers `(y1, y2, ..., yn)` for the same or similar prompt `x`
- Measure the **similarity or contradiction** between outputs

**Interpretation:**
- If all responses align closely → high confidence  
- If responses conflict or vary → low confidence

**Note:**  
LLMs can still be consistently wrong if not trained on a specific topic.

### RealWorld Example

If everyone gives the same answer, you're more confident it's right.  
If everyone gives different answers, your confidence drops.

Example responses:

- "Take the red line to Central Station."
- "You need a taxi, the trains are shut down."
- "The airport? You’ll need to change three buses."

That mismatch mirrors LLM inconsistency — if multiple generations contradict each other, it signals uncertainty.

---

## 2. Self-Reflection Certainty (SRC) — Intrinsic Evaluation

- Prompt the LLM to assess **its own answer** `y` for the input `x`

Methods:
- Ask: *"How likely is this answer to be correct?"* using multiple-choice categories (e.g., *Very Likely*, *Somewhat Likely*, *Unlikely*)
- Ask for a **justification** of the answer's correctness

**Notes:**
- Earlier versions using a 0–100 scale were too optimistic (LLMs rated themselves too highly)
- More reliable for:
  - Factual
  - Common sense
  - Algorithmic questions
- Less reliable for:
  - Abstract
  - Philosophical
  - Subjective queries

  ### RealWorld Example
  Now imagine if each person you asked also explained how confident they were:

- "I’m 100% sure — I live right by the airport."
- "I think this is right, but I’m not sure — I don’t take public transit often."

This is the LLM evaluating how confident it is in its own answer.

---

## Final Confidence Score

The overall confidence score is a combination of both OC and SRC:

```python
C(x, y) = f(ObservedConsistency(x, y), SelfReflectionCertainty(x, y))
Where f can be a:

Simple average, or

Weighted function (domain-specific)


## This Repository

This repository provides a Python3 implementation of the confidence scoring function based on Self-Reflection Certainty (SRC) as described in the paper.



## Usage