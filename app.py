import math
import random
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Page Config & Session Init
# =========================

st.set_page_config(
    page_title="AI Stats Tutor",
    page_icon="📊",
    layout="wide",
)

if "messages" not in st.session_state:
    # chat history: list of {"role": "user" / "assistant", "content": str}
    st.session_state.messages: List[Dict[str, str]] = []

if "current_topic" not in st.session_state:
    st.session_state.current_topic = "Probability"

if "practice_state" not in st.session_state:
    st.session_state.practice_state: Dict[str, Any] = {
        "problem": None,
        "hint_used": False,
        "correct_count": 0,
        "total_count": 0,
    }


# =========================
# Backend / AI STUBS
# (Person 1 will replace these)
# =========================

def get_tutor_response(user_msg: str, topic: str, difficulty: int) -> str:
    """
    TODO (Person 1): Replace this stub with real LLM / backend call.
    For now, it returns a simple, topic-aware canned explanation.
    """
    base = {
        "Probability": "Think about probability as the long-run frequency of an event.",
        "Hypothesis Testing": "In hypothesis testing, we compare data to a null hypothesis using a test statistic.",
        "Regression": "In regression, we model how a response variable changes with one or more predictors.",
    }.get(topic, "")

    return (
        f"Great question about **{topic}**!\n\n"
        f"You said: “{user_msg}”.\n\n"
        f"{base}\n\n"
        f"(This is a placeholder response. The real tutor logic will be added by Person 1.)"
    )


def generate_problem(topic: str, difficulty: int) -> Dict[str, Any]:
    """
    TODO (Person 1): Replace with dynamic / LLM-powered problem generation.

    For now, we generate very simple deterministic problems that:
      - depend on topic
      - encode the correct answer as a string
    """
    if topic == "Probability":
        # coin flip / basic probability
        n_flips = 5 + difficulty  # small scaling with difficulty
        question = (
            f"You flip a fair coin {n_flips} times.\n"
            f"What is the probability of getting exactly 2 heads? "
            f"(Give your answer as a rounded decimal with 3 digits, e.g. 0.312)"
        )
        # compute answer
        from math import comb

        p = (comb(n_flips, 2) * (0.5 ** n_flips))
        correct_answer = f"{p:.3f}"
        hint = (
            "Use the binomial formula: C(n, k) * p^k * (1-p)^(n-k) with n = "
            f"{n_flips}, k = 2, and p = 0.5."
        )

    elif topic == "Hypothesis Testing":
        question = (
            "You test H₀: μ = 100 vs H₁: μ ≠ 100 with n = 36, "
            "sample mean = 104, and known σ = 12. "
            "Compute the z-statistic (round to 2 decimals)."
        )
        # z = (x̄ - μ0) / (σ / sqrt(n))
        z = (104 - 100) / (12 / math.sqrt(36))
        correct_answer = f"{z:.2f}"  # 2.00
        hint = "Use z = (x̄ − μ₀) / (σ / √n). Here x̄ = 104, μ₀ = 100, σ = 12, n = 36."

    else:  # Regression or default
        question = (
            "In a simple linear regression, the estimated slope is 1.5.\n"
            "Interpret the slope in one sentence assuming x is 'hours studied' "
            "and y is 'exam score'."
        )
        correct_answer = (
            "For each additional hour studied, the exam score increases by about 1.5 points."
        )
        hint = (
            "Think: for a 1-unit increase in x (hours studied), how much does y "
            "change on average according to the model?"
        )

    return {
        "topic": topic,
        "difficulty": difficulty,
        "question": question,
        "correct_answer": correct_answer,
        "hint": hint,
    }


def check_answer(user_answer: str, correct_answer: str) -> bool:
    """
    Very basic answer checking:
      - if the correct_answer looks numeric, compare numerically with tolerance
      - otherwise, do a fuzzy string check (case-folded substring match).
    """
    # numeric?
    try:
        ca = float(correct_answer)
        ua = float(user_answer)
        return abs(ca - ua) < 0.01  # tolerance
    except Exception:
        # text comparison
        return correct_answer.lower() in user_answer.lower()


def get_hint(problem: Dict[str, Any]) -> str:
    """
    TODO (Person 1): Could be replaced with LLM-based hints.
    """
    return problem.get("hint", "Try breaking the problem into smaller steps.")


# =========================
# Visualization Helpers
# =========================

def show_probability_visuals():
    st.subheader("🎲 Sampling from a Binomial Distribution")

    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Number of trials (n)", 5, 100, 20, step=5)
        p = st.slider("Probability of success (p)", 0.05, 0.95, 0.5, step=0.05)

    with col2:
        num_samples = st.slider("Number of simulated experiments", 100, 5000, 1000, step=100)

    # simulate
    data = np.random.binomial(n=n, p=p, size=num_samples)

    st.write(
        f"We simulate **{num_samples}** experiments, each with **n = {n}** trials, "
        f"and success probability **p = {p:.2f}**."
    )

    fig, ax = plt.subplots()
    ax.hist(data, bins=range(0, n + 2), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of successes")
    ax.set_ylabel("Frequency")
    ax.set_title("Empirical Distribution of Successes (Binomial Simulation)")
    st.pyplot(fig)


def show_clt_visual():
    st.subheader("📈 Central Limit Theorem (CLT) Demo")

    col1, col2 = st.columns(2)
    with col1:
        dist = st.selectbox("Base distribution", ["Uniform(0,1)", "Exponential(λ=1)"])
    with col2:
        sample_size = st.slider("Sample size per experiment (n)", 2, 200, 30, step=2)

    num_experiments = st.slider("Number of experiments", 100, 5000, 2000, step=100)

    if dist == "Uniform(0,1)":
        base_samples = np.random.uniform(0, 1, size=(num_experiments, sample_size))
    else:
        base_samples = np.random.exponential(1.0, size=(num_experiments, sample_size))

    sample_means = base_samples.mean(axis=1)

    st.write(
        f"We draw **{num_experiments}** samples, each of size **n = {sample_size}**, "
        f"from a **{dist}** distribution and plot the distribution of sample means."
    )

    fig, ax = plt.subplots()
    ax.hist(sample_means, bins=40, edgecolor="black", alpha=0.7, density=True)
    ax.set_xlabel("Sample mean")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Sample Means (CLT in action)")
    st.pyplot(fig)


def show_regression_visual():
    st.subheader("📉 Simple Linear Regression Demo")

    n_points = st.slider("Number of data points", 10, 200, 50, step=5)
    true_slope = st.slider("True slope", 0.5, 3.0, 1.5, step=0.1)
    true_intercept = st.slider("True intercept", -5.0, 5.0, 0.0, step=0.5)
    noise_std = st.slider("Noise (standard deviation)", 0.5, 5.0, 2.0, step=0.5)

    # generate data
    x = np.linspace(0, 10, n_points)
    noise = np.random.normal(0, noise_std, size=n_points)
    y = true_intercept + true_slope * x + noise

    # fit line
    slope_hat, intercept_hat = np.polyfit(x, y, 1)

    df = pd.DataFrame({"x": x, "y": y})

    st.write(
        f"True model: y = {true_intercept:.2f} + {true_slope:.2f}·x  \n"
        f"Estimated model: y = {intercept_hat:.2f} + {slope_hat:.2f}·x"
    )

    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], alpha=0.7, label="Data points")
    ax.plot(x, intercept_hat + slope_hat * x, label="Fitted line", linewidth=2)
    ax.set_xlabel("x (e.g., hours studied)")
    ax.set_ylabel("y (e.g., exam score)")
    ax.set_title("Regression: Fitted Line vs Data")
    ax.legend()
    st.pyplot(fig)


def show_hypothesis_visual():
    st.subheader("🔍 Hypothesis Test (z-test) Visual")

    mu0 = st.number_input("Null mean (μ₀)", value=100.0)
    sigma = st.number_input("Known σ", value=15.0)
    n = st.number_input("Sample size (n)", value=36.0, step=1.0)
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, step=0.01)
    x_bar = st.number_input("Observed sample mean (x̄)", value=108.0)

    # compute z and critical values
    z = (x_bar - mu0) / (sigma / math.sqrt(n))
    from scipy.stats import norm  # Person 1 can also remove if not allowed

    z_crit = norm.ppf(1 - alpha / 2)
    st.write(f"Computed z-statistic: **{z:.2f}**")
    st.write(f"Critical values for two-sided test: **±{z_crit:.2f}**")

    # plot standard normal with rejection regions
    xs = np.linspace(-4, 4, 400)
    ys = norm.pdf(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label="Standard Normal PDF")

    # shading rejection regions
    ax.fill_between(xs, 0, ys, where=(xs <= -z_crit), alpha=0.3, label="Rejection region")
    ax.fill_between(xs, 0, ys, where=(xs >= z_crit), alpha=0.3)

    # observed z
    ax.axvline(z, color="black", linestyle="--", label="Observed z")

    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.set_title("Hypothesis Test Rejection Regions")
    ax.legend()
    st.pyplot(fig)


# =========================
# Layout Components
# =========================

def render_sidebar():
    st.sidebar.title("AI Stats Tutor")

    topic = st.sidebar.radio(
        "Choose a topic",
        ["Probability", "Hypothesis Testing", "Regression"],
        index=["Probability", "Hypothesis Testing", "Regression"].index(
            st.session_state.current_topic
        ),
    )
    st.session_state.current_topic = topic

    difficulty = st.sidebar.slider("Difficulty", 1, 5, 2)
    st.sidebar.markdown(
        "_Difficulty controls how challenging the practice problems will be._"
    )

    if st.sidebar.button("Reset session"):
        st.session_state.messages = []
        st.session_state.practice_state = {
            "problem": None,
            "hint_used": False,
            "correct_count": 0,
            "total_count": 0,
        }
        st.sidebar.success("Session reset!")

    return topic, difficulty


def tutor_chat_ui(topic: str, difficulty: int):
    st.subheader("💬 Tutor Chat")

    # show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    user_input = st.chat_input("Ask a question about this topic…")
    if user_input:
        # store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # get tutor response (stub for now)
        response = get_tutor_response(user_input, topic, difficulty)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # display assistant message immediately
        with st.chat_message("assistant"):
            st.markdown(response)


def visual_demos_ui():
    st.subheader("📊 Visual Demos")

    viz_option = st.selectbox(
        "Choose a visualization",
        [
            "Probability: Binomial Sampling",
            "Central Limit Theorem",
            "Regression: Fitted Line",
            "Hypothesis Testing: z-test",
        ],
    )

    if viz_option == "Probability: Binomial Sampling":
        show_probability_visuals()
    elif viz_option == "Central Limit Theorem":
        show_clt_visual()
    elif viz_option == "Regression: Fitted Line":
        show_regression_visual()
    else:
        show_hypothesis_visual()


def practice_ui(topic: str, difficulty: int):
    st.subheader("📝 Practice Problems")

    ps = st.session_state.practice_state

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct", ps["correct_count"])
    with col2:
        st.metric("Total Attempted", ps["total_count"])
    with col3:
        accuracy = (
            f"{(ps['correct_count'] / ps['total_count']) * 100:.1f}%"
            if ps["total_count"] > 0
            else "—"
        )
        st.metric("Accuracy", accuracy)

    st.markdown("---")

    # generate problem if none or user clicks "New problem"
    if ps["problem"] is None or st.button("🆕 New problem"):
        ps["problem"] = generate_problem(topic, difficulty)
        ps["hint_used"] = False

    problem = ps["problem"]
    st.markdown(f"**Topic:** {problem['topic']}  \n**Difficulty:** {problem['difficulty']}")
    st.markdown("#### Problem")
    st.write(problem["question"])

    # answer input
    answer = st.text_input("Your answer:")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        submit_clicked = st.button("Submit answer")
    with col_btn2:
        hint_clicked = st.button("Get hint")

    if hint_clicked:
        hint_text = get_hint(problem)
        ps["hint_used"] = True
        st.info(f"💡 Hint: {hint_text}")

    if submit_clicked and answer.strip():
        ps["total_count"] += 1
        if check_answer(answer, problem["correct_answer"]):
            ps["correct_count"] += 1
            st.success("✅ Nice! That looks correct.")
        else:
            st.error(
                f"❌ Not quite. One expected answer was: `{problem['correct_answer']}` "
                "(answers close to this may also be considered correct)."
            )


# =========================
# Main App
# =========================

def main():
    topic, difficulty = render_sidebar()

    st.title("📚 AI Tutor: Statistics for Data Science")
    st.markdown(
        f"""
Welcome! This tutor focuses on **{topic}**.

Use the tabs below to:
- Chat with the tutor for step-by-step explanations
- Explore visual demos for intuition
- Practice problems with hints and feedback
"""
    )

    tab_chat, tab_viz, tab_practice = st.tabs(
        ["💬 Tutor Chat", "📊 Visual Demos", "📝 Practice"]
    )

    with tab_chat:
        tutor_chat_ui(topic, difficulty)

    with tab_viz:
        visual_demos_ui()

    with tab_practice:
        practice_ui(topic, difficulty)


if __name__ == "__main__":
    main()
