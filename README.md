# InsuranceBOT

This repository contains the complete source code for my submission to the InsureBot Quest 2025 hackathon. "Veena" is a conversational voice AI designed to function as an outbound calling agent for ValuEnable.

## Features

- **Low Latency:** Achieves an average response time of 2-4 seconds.
- **High Accuracy:** Uses a RAG architecture grounded in the provided hackathon documents.
- **Natural Interaction:** Features a clear female voice, dynamic speech pacing, and full interruption handling.
- **Stateful Conversation:** A state machine prevents the bot from getting stuck in conversational loops.

## How to Run the Project

### Prerequisites

- Python 3.10+
- FFmpeg installed and added to your system's PATH.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-github-repo-link]
    cd [your-project-folder]
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    - Rename the `.env.example` file to `.env`.
    - Open the `.env` file and add your Groq API key: `GROQ_API_KEY=your_key_here`

5.  **Run the application:**
    ```bash
    python app.py
    ```

6.  Open your web browser and navigate to **`http://127.0.0.1:5001`**.

### Project Structure

- `app.py`: The main Flask backend server.
- `index.html`: The single-page HTML/CSS/JS frontend.
- `requirements.txt`: A list of all necessary Python libraries.
- `hackathon_data/`: Contains all the source material provided by the hackathon (script, knowledge base, recordings).
- `cache/`: (Generated automatically) Stores the pre-processed RAG index for fast startups.
