<div align="center">

# 🤖 InsureBot Quest 2025: "Veena" 🤖

### An AI-Powered Outbound Calling Agent for ValuEnable

_Submission by The LoneWolf_

</div>

---

> **Project Goal:** To develop "Veena," a conversational voice AI that can proactively contact customers, remind them of pending premium payments, and guide them toward resolution, strictly following a provided calling script.

<br>

<div align="center">

**[➡️ View Repo](https://github.com/Celest1al/InsuranceBOT/tree/main)** &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; **[📺 Watch the Demo](https://youtu.be/N-IUWELuUEw)**

</div>

---

## ✨ Key Features

-   **🗣️ Natural Conversation:** Simulates a human agent with a clear female voice, dynamic speech pacing, and full interruption handling.
-   **⚡ Low Latency:** Achieves an average response time of 2-4 seconds, enabling real-time, fluid conversation.
-   **🧠 High Accuracy:** Utilizes a RAG (Retrieval-Augmented Generation) architecture grounded in the hackathon's documents to provide factually correct answers.
-   **🤖 Stateful Logic:** A smart state machine prevents the bot from getting stuck in conversational loops, ensuring it follows the script logically.
-   **🌐 Multi-Language Support:** Capable of switching to Hindi or Marathi based on user preference.

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=LangChain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/Groq-000000?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"/>
  <img src="https://img.shields.io/badge/Faiss-4A90E2?style=for-the-badge&logo=faiss&logoColor=white" alt="Faiss"/>
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5"/>
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3"/>
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript"/>
</p>

---

## 🚀 How to Run the Project

### Prerequisites

-   Python 3.10+
-   FFmpeg installed and added to your system's PATH.

### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Celest14l/InsuranceBOT.git](https://github.com/Celest14l/InsuranceBOT.git)
    cd InsuranceBOT
    ```

2.  **Create and Activate a Virtual Environment** (Recommended):
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Key:**
    -   Rename the `.env.example` file to `.env`.
    -   Open the `.env` file and add your Groq API key: `GROQ_API_KEY=your_key_here`

5.  **Run the Application:**
    ```bash
    python app.py
    ```

6.  **Launch the Frontend:**
    -   Open your web browser and navigate to **`http://127.0.0.1:5001`**.
