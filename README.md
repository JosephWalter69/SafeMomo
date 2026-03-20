#SafeMomo
# Ghana MoMo Fraud Guard 🛡️

**Final Year Computer Science Project**  
Protecting mobile money users in Ghana from SMS scams, phishing links, and suspicious transactions.

**Live Demo:** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

### Features
- **SMS Check** – Detects fake/scam messages using DistilBERT (fine-tuned on Ghana-specific data)
- **Link Check** – Identifies phishing URLs using XGBoost trained on malicious URL dataset
- **Transaction Cross Check** – Flags suspicious patterns using Isolation Forest on PaySim-like data
- **User Feedback** – Thumbs up/down to help improve the system over time
- **Feature Phone Support** – Simulation of SMS/USSD verification
- **History** – View past checks

### Technologies
- Streamlit (web app)
- Hugging Face (model hosting)
- XGBoost & scikit-learn (ML)
- Pandas & NumPy (data handling)

### How to Run Locally
```bash
git clone https://github.com/JosephWalter69/SafeMomo.git
cd SafeMomo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
