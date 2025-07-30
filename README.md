# 📄 Insurance Analyst AI Chatbot

This is an LLM-powered QA chatbot built with Streamlit, LangChain, and Google's Gemini Pro. It answers questions about insurance policy documents.

### Prerequisites
* Python 3.8+
* An active internet connection

### 🔑 Step 1: Get Your Google API Key
This application requires a Google API key to use the Gemini model.
1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Click "**Create API key**" and copy the key.

### ⚙️ Step 2: Setup and Installation
1.  Unzip the project folder.
2.  Open your terminal or command prompt and navigate into the project directory:
    ```bash
    cd path/to/"D:\baja_hackathon_project"

    ```
3.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  [cite_start]Install all the required packages from `requirements.txt`[cite: 2]:
    ```bash
    pip install -r requirements.txt
    ```
5.  Create a new file named `.env` in the main project folder. [cite_start]Open it and add your Google API key in the following format[cite: 1]:
    ```
    GOOGLE_API_KEY='YOUR_API_KEY_HERE'
    ```

### ▶️ Step 3: Run the Application
1.  Make sure you are in the main project directory in your terminal.
2.  Run the Streamlit application with the following command:
    ```bash
    streamlit run app.py
    ```
3.  A new tab should open in your web browser with the running application. Enjoy!