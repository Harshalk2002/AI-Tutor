# Math Tutor App – AI-Powered Learning Assistant

## 📌 1. Project Overview
The **Math Tutor App** is an AI-powered teaching assistant built to help students understand mathematical concepts through step-by-step explanations, worked examples, and interactive chat-based support.  
It integrates **React.js**, **FastAPI**, and **LLM-based reasoning**, deployed using **GitLab CI/CD**, following professor-approved project structure.

---

✨ **Try it live:**  
Explore the AI Math Tutor in action → https://aimathtutorgsu.streamlit.app/


## 🏗️ 2. System Architecture

```
┌──────────────────┐      ┌────────────────────┐      ┌────────────────────────┐
│   React Frontend  │ ---> │    FastAPI Backend │ ---> │       AI/LLM Engine     │
│ (User Interaction)│      │ (API + Routing)    │      │ (Reasoning + Memory)    │
└──────────────────┘      └────────────────────┘      └────────────────────────┘
         ▲                        │                               │
         └────────── Streamlit App (Demo + Tutor View) ───────────┘
```

---

## 🧩 3. Features
- AI-powered math explanations  
- Step-by-step problem solving  
- Clean React chat UI  
- FastAPI backend with modular API routes  
- Memory-driven learning  
- Auto-deployment using GitLab CI/CD  
- Streamlit demo interface  

---

## ⚙️ 4. Technical Challenges & Solutions

**Challenge 1 — Broken CI/CD Pipeline**  
Pipeline was failing due to unclean repo files and incorrect directory structure.  
✔ Cleaned the repository, matched professor's starter template, fixed variable and port mismatches.

**Challenge 2 — Frontend/Backend Connection Issues**  
React could not hit backend endpoints due to CORS and path mismatches.  
✔ Implemented proper CORS middleware and standardized API routes.

**Challenge 3 — Multi-step Math Reasoning**  
LLM needed structured reasoning instead of random output.  
✔ Built modular functions to force systematic explanations.

**Challenge 4 — GitLab Runner Environment Differences**  
Local environment worked but GitLab runner failed.  
✔ Updated folder paths, Docker config, and pipeline stages.

---

## 🚀 5. How to Run Locally

### 🔧 Backend (FastAPI)
```
cd fastapi_backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 🎨 Frontend (React.js)
```
cd react_js_app
npm install
npm start
```

### 📘 Streamlit Demo
```
cd streamlit_app
streamlit run app.py
```

---

## 📂 6. Folder Structure

```
Math-Tutor-App/
│── react_js_app/
│   └── src/
│── fastapi_backend/
│   ├── main.py
│   ├── requirements.txt
│── streamlit_app/
│   ├── app.py
│── .gitlab-ci.yml
│── README.md
```

---

## 🧠 7. Memory System

### Short-Term Memory
- Tracks session context  
- Supports multi-step problem solving  

### Long-Term Memory
- Remembers user difficulty patterns  
- Enables personalized tutoring  

---

## 🧪 8. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/solve` | Solve a math problem step-by-step |
| POST | `/explain` | Explain a math concept |
| GET | `/health` | Health check |

---

## 🖼️ 9. Screenshots (Placeholders)

Add your screenshots inside `/screenshots/`.

```
/screenshots
  ├── homepage.png
  ├── chat_example.png
  ├── explanation_view.png
```

---

## ⭐ Final Notes
This project demonstrates full-stack engineering, prompt engineering, agentic reasoning, and CI/CD deployment best practices.  
If helpful, please ⭐ star the repository.

