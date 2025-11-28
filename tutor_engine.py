"""
tutor_engine.py

Backend logic for the AI Stats Tutor:
- LLM-powered tutor responses (with fallback)
- Problem generation
- Hint generation
- Answer checking
"""

import math
import os
import random
from typing import Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# -------------------------------------------------------
# Load your .env file (LOCAL ONLY)
# -------------------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

import streamlit as st

class TutorEngine:
    def __init__(self):
        # Load API key depending on environment
        api_key = None

        # 1. Streamlit Cloud (secrets)
        if hasattr(st, "secrets"):
            api_key = st.secrets.get("OPENAI_API_KEY")

        # 2. Local development (.env)
        if not api_key:
            env_path = Path(__file__).resolve().parent / ".env"
            load_dotenv(dotenv_path=env_path)
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                self.client = None
                print("ERROR INITIALIZING OPENAI CLIENT:", e)
        else:
            print("NO OPENAI_API_KEY FOUND")
            self.client = None

class TutorEngine:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                print("OpenAI client initialized.")
            except Exception as e:
                self.client = None
                print("ERROR INITIALIZING OPENAI CLIENT:", e)
        else:
            print("NO OPENAI_API_KEY FOUND")
            self.client = None

    # -------------------------------
    # LLM Tutor Response
    # -------------------------------
    def get_tutor_response(self, user_msg: str, topic: str, difficulty: int) -> str:
        # If LLM available, use it
        if self.client is not None:
            try:
                return self._llm_response(user_msg, topic, difficulty)
            except Exception as e:
                return f"(LLM Error: {e})\n\n" + self._fallback_explanation(
                    user_msg, topic, difficulty
                )

        # otherwise fallback
        return self._fallback_explanation(user_msg, topic, difficulty)

    def _llm_response(self, user_msg: str, topic: str, difficulty: int) -> str:
        """
        Uses OpenAI chat completion to answer as an AI statistics tutor.
        """

        difficulty_text = {
            1: "Explain like I'm completely new to statistics. Use very simple language and intuition.",
            2: "Explain step-by-step with simple math and minimal notation.",
            3: "Explain clearly with some formulas and definitions.",
            4: "Assume I know basic stats; focus on deeper reasoning and connections.",
            5: "Assume I'm an advanced stats student; you can be concise and technical.",
        }.get(difficulty, "Explain in a clear and student-friendly way.")

        system_prompt = f"""
You are an AI tutor helping a student in a university-level **Statistics for Data Science** course.

Your job:
- Focus on the topic: {topic}
- {difficulty_text}
- Use examples related to data science, machine learning, A/B testing, or real analytics when helpful.
- NEVER just give final answers to graded-looking questions; instead, guide step by step.
- Do NOT hallucinate formulas. If you're unsure, say so and give intuition instead.
- Keep the tone encouraging and supportive.
"""

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",  # or any compatible model you have access to
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )

        answer = resp.choices[0].message.content
        return answer

    # -------------------------------
    # Fallback explanation (rule-based)
    # -------------------------------
    def _fallback_explanation(self, user_msg: str, topic: str, difficulty: int) -> str:
        topic_intro = {
            "Probability": (
                "Probability quantifies how likely an event is. "
                "For example, the probability of getting heads on a fair coin is 0.5."
            ),
            "Hypothesis Testing": (
                "In hypothesis testing, we start with a null hypothesis H₀, "
                "compute a test statistic, and decide how unusual our sample is under H₀."
            ),
            "Regression": (
                "Regression models how a response variable changes with one predictor. "
                "In simple linear regression, we estimate a line y = β₀ + β₁x + ε."
            ),
        }.get(topic, "")

        difficulty_text = {
            1: "I'll keep the explanation very basic and intuitive.",
            2: "I'll explain step-by-step with simple math.",
            3: "I'll include a bit more detail and notation.",
            4: "I'll assume some familiarity with formulas and focus on reasoning.",
            5: "I'll use more advanced terminology and connect multiple concepts.",
        }.get(difficulty, "I'll adjust the detail level to your understanding.")

        return (
            f"**Topic:** {topic}\n\n"
            f"You asked:\n> {user_msg}\n\n"
            f"{difficulty_text}\n\n"
            f"{topic_intro}\n\n"
            f"_Note: This is a fallback explanation. "
            f"If an LLM is configured, you will receive a richer explanation._"
        )

    # =========================
    # Problem Generation
    # =========================

    def generate_problem(self, topic: str, difficulty: int) -> Dict[str, Any]:
        """
        Creates a practice problem with:
        - question (str)
        - correct_answer (str)
        - hint (str)
        - topic, difficulty metadata
        """

        if topic == "Probability":
            return self._generate_probability_problem(difficulty)
        elif topic == "Hypothesis Testing":
            return self._generate_hypothesis_problem(difficulty)
        else:  # Regression or any other
            return self._generate_regression_problem(difficulty)

    def _generate_probability_problem(self, difficulty: int) -> Dict[str, Any]:
        # scale complexity with difficulty
        n_flips = random.choice([5, 6, 7, 8]) + difficulty
        k_heads = random.randint(1, min(3 + difficulty, n_flips))

        question = (
            f"You flip a fair coin **{n_flips}** times.\n"
            f"What is the probability of getting exactly **{k_heads}** heads?\n"
            f"Give your answer as a rounded decimal with 3 digits (e.g., 0.312)."
        )

        from math import comb

        p = (comb(n_flips, k_heads) * (0.5 ** n_flips))
        correct_answer = f"{p:.3f}"

        hint = (
            "Use the binomial formula: C(n, k) · p^k · (1−p)^(n−k) "
            f"with n = {n_flips}, k = {k_heads}, and p = 0.5."
        )

        return {
            "topic": "Probability",
            "difficulty": difficulty,
            "question": question,
            "correct_answer": correct_answer,
            "hint": hint,
        }

    def _generate_hypothesis_problem(self, difficulty: int) -> Dict[str, Any]:
        # simple z-test scenario
        mu0 = random.choice([50, 100])
        sigma = random.choice([8, 10, 12, 15])
        n = random.choice([25, 36, 49])
        x_bar_shift = random.choice([2, 4, 6])
        x_bar = mu0 + x_bar_shift

        question = (
            f"You test H₀: μ = {mu0} vs H₁: μ ≠ {mu0} with:\n"
            f"- Sample size n = {n}\n"
            f"- Sample mean x̄ = {x_bar}\n"
            f"- Known standard deviation σ = {sigma}\n\n"
            f"Compute the **z-statistic** and round to 2 decimal places."
        )

        z = (x_bar - mu0) / (sigma / math.sqrt(n))
        correct_answer = f"{z:.2f}"

        hint = (
            "Recall: z = (x̄ − μ₀) / (σ / √n). "
            f"Plug in x̄ = {x_bar}, μ₀ = {mu0}, σ = {sigma}, n = {n}."
        )

        return {
            "topic": "Hypothesis Testing",
            "difficulty": difficulty,
            "question": question,
            "correct_answer": correct_answer,
            "hint": hint,
        }

    def _generate_regression_problem(self, difficulty: int) -> Dict[str, Any]:
        slope = random.choice([0.8, 1.2, 1.5, 2.0])
        intercept = random.choice([-5, 0, 10])
        context_x = "hours studied"
        context_y = "exam score"

        patterns = [
            (
                f"In a simple linear regression, the fitted model is:\n"
                f"**ŷ = {intercept:.1f} + {slope:.1f}·x**, where x is `{context_x}` "
                f"and y is `{context_y}`.\n\n"
                f"Interpret the slope in a single, clear sentence."
            ),
            (
                f"In a simple regression of `{context_y}` on `{context_x}`, "
                f"the estimated slope coefficient is **{slope:.1f}**.\n"
                f"What does this mean in context?"
            ),
        ]

        question = random.choice(patterns)
        correct_answer = (
            f"For each additional 1 unit of {context_x}, the {context_y} increases "
            f"on average by about {slope:.1f} points."
        )

        hint = (
            "Think: the slope tells you the change in predicted y "
            "for a 1-unit increase in x, holding everything else constant."
        )

        return {
            "topic": "Regression",
            "difficulty": difficulty,
            "question": question,
            "correct_answer": correct_answer,
            "hint": hint,
        }

    # =========================
    # Answer & Hint Helpers
    # =========================

    def check_answer(self, user_answer: str, correct_answer: str) -> bool:
        """
        For numeric answers: compare with tolerance.
        For text answers: do a simple case-insensitive substring check.
        """
        user_answer = user_answer.strip()
        if not user_answer:
            return False

        # numeric comparison
        try:
            ca = float(correct_answer)
            ua = float(user_answer)
            return abs(ca - ua) < 0.01  # 0.01 tolerance
        except Exception:
            # text comparison
            return correct_answer.lower() in user_answer.lower()

    def get_hint(self, problem: Dict[str, Any]) -> str:
        """
        Returns a hint for the given problem.
        """
        return problem.get("hint", "Try breaking the problem into smaller steps.")
