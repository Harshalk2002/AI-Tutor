import math
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tutor_engine import TutorEngine  # backend logic


st.set_page_config(
    page_title="AI Stats Tutor",
    page_icon="📊",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 I'm your **AI Stats Tutor**.\n\n"
                "You can ask me things like:\n"
                "- “Explain the Central Limit Theorem with an example.”\n"
                "- “What does a p-value actually mean?”\n"
                "- “How do I interpret the slope in regression?”"
            ),
        }
    ]

if "current_topic" not in st.session_state:
    st.session_state.current_topic = "Probability"

if "practice_state" not in st.session_state:
    st.session_state.practice_state: Dict[str, Any] = {
        "problem": None,
        "hint_used": False,
        "correct_count": 0,
        "total_count": 0,
    }

if "engine" not in st.session_state:
    st.session_state.engine = TutorEngine()

engine: TutorEngine = st.session_state.engine


def show_probability_visuals():
    st.subheader("🎲 Sampling from a Binomial Distribution")

    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Number of trials (n)", 5, 100, 20, step=5)
        p = st.slider("Probability of success (p)", 0.05, 0.95, 0.5, step=0.05)

    with col2:
        num_samples = st.slider(
            "Number of simulated experiments", 100, 5000, 1000, step=100
        )

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

    num_experiments = st.slider(
        "Number of experiments", 100, 5000, 2000, step=100
    )

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

    x = np.linspace(0, 10, n_points)
    noise = np.random.normal(0, noise_std, size=n_points)
    y = true_intercept + true_slope * x + noise

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

    z = (x_bar - mu0) / (sigma / math.sqrt(n))

    try:
        from scipy.stats import norm
    except ImportError:
        st.error("scipy is not installed. Please add 'scipy' to requirements.txt.")
        return

    z_crit = norm.ppf(1 - alpha / 2)
    st.write(f"Computed z-statistic: **{z:.2f}**")
    st.write(f"Critical values for two-sided test: **±{z_crit:.2f}**")

    xs = np.linspace(-4, 4, 400)
    ys = norm.pdf(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label="Standard Normal PDF")
    ax.fill_between(xs, 0, ys, where=(xs <= -z_crit), alpha=0.3, label="Rejection region")
    ax.fill_between(xs, 0, ys, where=(xs >= z_crit), alpha=0.3)
    ax.axvline(z, color="black", linestyle="--", label="Observed z")

    ax.set_xlabel("z")
    ax.set_ylabel("Density")
    ax.set_title("Hypothesis Test Rejection Regions")
    ax.legend()
    st.pyplot(fig)


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

    # 1) Render full chat history (oldest → newest) ABOVE the input
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 2) Chat input always at the bottom
    user_input = st.chat_input("Type your question here…")

    # 3) When user sends a message, update history ONLY (no extra rendering here)
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get assistant response from the engine
        response = engine.get_tutor_response(user_input, topic, difficulty)

        # Add assistant message to history
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )



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

    if ps["problem"] is None or st.button("🆕 New problem"):
        ps["problem"] = engine.generate_problem(topic, difficulty)
        ps["hint_used"] = False

    problem = ps["problem"]
    st.markdown(
        f"**Topic:** {problem['topic']}  \n**Difficulty:** {problem['difficulty']}"
    )
    st.markdown("#### Problem")
    st.write(problem["question"])

    answer = st.text_input("Your answer:")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        submit_clicked = st.button("Submit answer")
    with col_btn2:
        hint_clicked = st.button("Get hint")

    if hint_clicked:
        hint_text = engine.get_hint(problem)
        ps["hint_used"] = True
        st.info(f"💡 Hint: {hint_text}")

    if submit_clicked and answer.strip():
        ps["total_count"] += 1
        if engine.check_answer(answer, problem["correct_answer"]):
            ps["correct_count"] += 1
            st.success("✅ Nice! That looks correct.")
        else:
            st.error(
                f"❌ Not quite. One expected answer was: `{problem['correct_answer']}` "
                "(answers close to this may also be considered correct)."
            )


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
