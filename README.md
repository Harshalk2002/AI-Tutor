# VisaJourney Agent: AI-Driven Compliance Assistant for F‑1 Visa Holders

## 📌 Overview  
VisaJourney Agent is a two-agent AI system designed to simplify complex U.S. immigration rules for F‑1 visa students. Instead of manually reading dense USCIS/DHS documents, this system extracts policies automatically and generates personalized timelines and compliance checklists.

This project was developed as part of **MSA 8770 – Text Analytics (Fall 2025)** at **Georgia State University**.

---

## 🧠 System Architecture

### **Agent 1 – Policy Extraction Agent**  
- Processes unstructured visa/immigration text (USCIS, DHS, SEVIS guidelines).  
- Converts legal language into structured rules.  
- Identifies risk levels, deadlines, and mandatory actions.  
- Creates short-term memory (per-document summaries) and long-term memory (accumulated rule base across many documents).

### **Agent 2 – Visa Journey Personalization Agent**  
- Uses the structured rules from Agent 1.  
- Generates timeline-based action plans for CPT, OPT, STEM OPT, SEVIS reporting, job-loss notifications, etc.  
- Provides personalized checklists based on the student profile (program dates, job status, employment type).  
- Supports month‑by‑month and week‑by‑week compliance reminders.

---

## 🎯 Key Features
- Automated extraction of immigration requirements  
- Personalized visa compliance timelines  
- Dynamic rule-based reasoning  
- Memory system (short‑term for current query + long‑term stored rules)  
- Easily deployable backend using FastAPI or Streamlit  
- Modular backend (`agents_backend.py`) for production use  

---

## 🏗️ Project Structure
```
/VisaJourney-Agent
│── agents_backend.py        # Core logic for timelines & rule processing
│── data/                    # Raw and cleaned guidelines (optional)
│── results/                 # Generated outputs
│── README.md                # Documentation
│── requirements.txt         # Install dependencies
│── streamlit_app.py         # (Optional) UI layer
│── fastapi_app.py           # (Optional) API service
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/VisaJourney-Agent.git
cd VisaJourney-Agent
```

### 2. Create Virtual Environment  
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App  
```bash
streamlit run streamlit_app.py
```

### 5. Run FastAPI Backend  
```bash
uvicorn fastapi_app:app --reload
```

---

## 🎓 Use Cases
- International students tracking CPT/OPT eligibility  
- University advisors assisting with SEVIS compliance  
- Automated reminder systems for deadlines  
- AI‑powered visa explanation chatbots  

---

## 👥 Team Members  
- **Abhay Prabhakar**  
- **Harshal Kamble**  
- **Pavithra Kannan**  
- **Jenny Nguyen**  
- **Jared Jones**

Instructor: **Dr. Soleymani**

---

## 📄 License  
This project is for academic use only. Not intended as legal advice.

---

## ⭐ Contributions  
Feel free to open PRs or suggestions.  
If this project helps you, please ⭐ star the repo!

