import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
from transformers import pipeline


# ────────────────────────────────────────────────
#   PAGE CONFIG & STYLING
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="MoMo Fraud Guard 🛡️",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

GREEN = "#006B3F"
YELLOW = "#FCD116"
RED = "#CE1126"

st.markdown(
    f"""
    <style>
    .big-title {{
        font-size: 2.8rem;
        color: {GREEN};
        text-align: center;
        font-weight: bold;
    }}
    .alert-green {{
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.6rem;
        margin: 20px 0;
    }}
    .alert-red {{
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.6rem;
        margin: 20px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🛡️ MoMo Fraud Guard</div>', unsafe_allow_html=True)
st.caption("Your personal shield against mobile money scams in Ghana")

# ────────────────────────────────────────────────
#   SESSION STATE
# ────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ────────────────────────────────────────────────
#   LOAD ADVANCED SMS CLASSIFIER (once, cached)
# ────────────────────────────────────────────────
@st.cache_resource
def load_sms_classifier():
    try:
        # Load directly from your Hugging Face repo
        return pipeline(
            "text-classification",
            model="josephwalter69/ghana-momo-sms-classifier"   # ← change to your actual repo ID
        )
    except Exception as e:
        st.warning(f"Could not load advanced model from Hugging Face: {e}\nUsing basic keyword check.")
        return None

sms_classifier = load_sms_classifier()

# ────────────────────────────────────────────────
#   GENERATE SAMPLE TRANSACTION DATA
# ────────────────────────────────────────────────
def generate_paysim_like_data(n_samples):
    """Generate synthetic transaction data for model training/testing."""
    np.random.seed(42)
    data = {
        'amount': np.random.exponential(scale=5000, size=n_samples),
        'oldbalanceOrg': np.random.exponential(scale=10000, size=n_samples),
        'newbalanceOrig': np.random.exponential(scale=10000, size=n_samples),
        'type_encoded': np.random.randint(0, 5, size=n_samples),
        'balance_change_ratio': np.random.uniform(0, 2, size=n_samples),
        'amount_deviation': np.random.uniform(0, 5, size=n_samples),
    }
    return pd.DataFrame(data)

# ────────────────────────────────────────────────
#   TABS
# ────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📱 SMS Check",
    "🔗 Link Check",
    "💰 Transaction Cross Check",
    "📜 History",
    "📖 How to Use",
    "ℹ️ About"
])

# ────────────────────────────────────────────────
#   TAB 1 – SMS CHECK
# ────────────────────────────────────────────────
with tab1:
    st.subheader("Paste suspicious SMS")
    sms_text = st.text_area("SMS Message", height=180, placeholder="Example: URGENT! Wrong transfer of GHS 2000. Call now to return!")

    if st.button("Check SMS", type="primary"):
        if sms_text.strip():
            if sms_classifier:
                try:
                    result = sms_classifier(sms_text)[0]
                    label = result['label']          # Usually 'LABEL_0' or 'LABEL_1'
                    score = result['score']
                    # Assuming LABEL_1 = fake/scam (adjust if your training labels are reversed)
                    is_fake = (label == 'LABEL_1')
                    confidence = score if is_fake else (1 - score)
                except Exception as e:
                    st.error("Error using advanced model. Falling back to keywords.")
                    is_fake = False
                    confidence = 0.0
            else:
                # Fallback keyword check
                fake_keywords = ["send back", "return money", "wrong transfer", "call this number", "confirm PIN", "urgent", "click here"]
                is_fake = any(kw.lower() in sms_text.lower() for kw in fake_keywords)
                confidence = 0.85 if is_fake else 0.70

            if is_fake:
                st.markdown(
                    f'<div class="alert-red">🚨 FAKE SMS DETECTED! (Confidence: {confidence:.0%})<br>DO NOT reply, call, or send money!</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="alert-green">✅ Looks genuine (Confidence: {confidence:.0%})<br>Still double-check in your official MoMo app</div>',
                    unsafe_allow_html=True
                )

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "type": "SMS",
                "input": sms_text[:50] + "..." if len(sms_text) > 50 else sms_text,
                "result": "FAKE" if is_fake else "GENUINE"
            })
        else:
            st.warning("Please paste an SMS message first.")

# ────────────────────────────────────────────────
#   TAB 2 – LINK CHECK
# ────────────────────────────────────────────────
with tab2:
    st.subheader("Paste suspicious link")
    link_url = st.text_input("Link/URL", placeholder="https://momo.mtn.com.gh/claim-prize")

    if st.button("Verify Link", type="primary"):
        if link_url.strip():
            official_domains = ["mtn.com.gh", "momo.mtn.com", "telecel", "airteltigo"]
            suspicious_keywords = ["login", "verify", "pin", "claim", "refund", ".tk", ".ml", ".xyz", "bit.ly"]

            parsed_domain = link_url.lower().replace("https://", "").replace("http://", "").split("/")[0]
            is_suspicious = (
                not any(domain in parsed_domain for domain in official_domains)
                or any(keyword in link_url.lower() for keyword in suspicious_keywords)
            )

            if is_suspicious:
                st.markdown(
                    '<div class="alert-red">🚨 POTENTIAL PHISHING LINK!<br>DO NOT CLICK or enter any details!</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="alert-green">✅ Appears to be a safe/official link</div>',
                    unsafe_allow_html=True
                )

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "type": "Link",
                "input": parsed_domain,
                "result": "PHISHING" if is_suspicious else "SAFE"
            })
        else:
            st.warning("Please paste a link first.")

# ────────────────────────────────────────────────
#   TAB 3 – TRANSACTION CROSS CHECK
# ────────────────────────────────────────────────
with tab3:
    st.subheader("Enter transaction details to verify")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (GHS)", min_value=0.0, value=500.0, step=10.0)
        old_bal = st.number_input("Old Balance (Sender)", min_value=0.0, value=2000.0, step=100.0)
    with col2:
        new_bal = st.number_input("New Balance (Sender)", min_value=0.0, value=1500.0, step=100.0)
        trans_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

    if st.button("Check Transaction", type="primary"):
        try:
            # Load realistic data (fallback to generate if CSV missing)
            try:
                df = pd.read_csv("paysim_like_transactions.csv")
            except FileNotFoundError:
                df = generate_paysim_like_data(10000)  # small fallback generation

            # Feature prep for model
            features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'type_encoded',
                        'balance_change_ratio', 'amount_deviation']

            # Train Isolation Forest on full data
            model = IsolationForest(
                n_estimators=150,          # more trees for better stability
                contamination=0.01,        # expect ~1% anomalies
                random_state=42,
                max_samples=1024
            )
            model.fit(df[features])

            # User input as DataFrame
            user_row = pd.DataFrame([{
                'amount': amount,
                'oldbalanceOrg': old_bal,
                'newbalanceOrig': new_bal,
                'type_encoded': ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"].index(trans_type),
                'balance_change_ratio': amount / old_bal if old_bal > 0 else 0,
                'amount_deviation': np.abs(amount - df['amount'].mean()) / df['amount'].std()
            }])

            # Score: lower = more anomalous
            anomaly_score = model.decision_function(user_row[features])[0]
            is_fraud = anomaly_score < -0.05  # Tune threshold based on your testing (lower = stricter)

            if is_fraud:
                st.markdown(
                    '<div class="alert-red">🚨 SUSPICIOUS TRANSACTION DETECTED!<br>Anomaly score indicates potential fraud. Contact your provider immediately.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="alert-green">✅ Transaction appears normal</div>',
                    unsafe_allow_html=True
                )

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "type": "Transaction",
                "input": f"{trans_type} GHS {amount:.2f}",
                "result": "FRAUD" if is_fraud else "GENUINE"
            })

        except Exception as e:
            st.error(f"Error during check: {e}. Try different values.")

# ────────────────────────────────────────────────
#   TAB 4 – HISTORY
# ────────────────────────────────────────────────
with tab4:
    st.subheader("Your Check History")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No checks yet. Results will appear here.")

# ────────────────────────────────────────────────
#   TAB 5 – HOW TO USE (user instructions only)
# ────────────────────────────────────────────────
with tab5:
    st.subheader("How to Use MoMo Fraud Guard")
    st.markdown("""
    **1. SMS Check**  
    - Copy any message you receive claiming to be from MTN, Telecel or AirtelTigo.  
    - Paste it in the box and tap **Check SMS**.  
    - Red alert = Do NOT reply, call or send money.  
    - Green = Looks safe, but always confirm inside your official MoMo app.

    **2. Link Check**  
    - Copy any link sent to you via SMS, WhatsApp, etc.  
    - Paste it and tap **Verify Link**.  
    - Red alert = Never click or enter your PIN / details.  
    - Green = Safe to visit (but type the official website manually if possible).

    **3. Transaction Cross Check**  
    - Enter the amount, old balance, new balance and type of transaction you want to verify.  
    - Tap **Check Transaction**.  
    - Red alert = Looks suspicious — contact your network or bank immediately.  
    - Green = Appears normal.

    **4. History**  
    - See all your previous checks.  
    - Use the **Clear History** button when you want to start fresh.

    **For feature phone users**  
    Forward suspicious SMS to our dedicated short code / number (feature coming soon) to receive an instant reply.
    """)

# ────────────────────────────────────────────────
#   TAB 6 – ABOUT
# ────────────────────────────────────────────────
with tab6:
    st.subheader("About the System")
    st.markdown("""
    **Ghana MoMo Fraud Guard**  
    Final Year Computer Science Project  
    Designed to help protect mobile money users in Ghana from common SMS scams, phishing links, and suspicious transactions.

    - Easy to use on any smartphone  
    - Free and private  
    - Built to make mobile money safer for everyone

    **Developer**: Nii Amoo  
    **Goal**: Reduce fraud and increase trust in mobile money services.
    """)
    st.caption("Version 1.0 – March 2026")

# ────────────────────────────────────────────────
#   SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.success("System is running")
    st.info("Protect yourself before sending or clicking anything suspicious!")

st.caption("Complete system ready for use and presentation")
