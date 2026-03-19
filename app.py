import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
from transformers import pipeline
import os
import csv

# ────────────────────────────────────────────────
#   CONFIG & STYLING
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
    .big-title {{font-size: 2.8rem; color: {GREEN}; text-align: center; font-weight: bold;}}
    .alert-green {{background-color: #d4edda; color: #155724; padding: 20px; border-radius: 15px; text-align: center; font-size: 1.6rem; margin: 20px 0;}}
    .alert-red   {{background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 15px; text-align: center; font-size: 1.6rem; margin: 20px 0;}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🛡️ MoMo Fraud Guard</div>', unsafe_allow_html=True)
st.caption("Your shield against mobile money scams in Ghana")

# ────────────────────────────────────────────────
#   SESSION STATE & FEEDBACK FILE
# ────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

FEEDBACK_FILE = "user_feedback.csv"

# Create feedback file with header if it doesn't exist
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "check_type", "input_text", "system_result", "user_label", "user_comment"])

def log_feedback(check_type, input_text, system_result, user_judgment, comment=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, check_type, input_text[:200], system_result, user_judgment, comment]
    
    with open(FEEDEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ────────────────────────────────────────────────
#   LOAD ADVANCED SMS CLASSIFIER
# ────────────────────────────────────────────────
@st.cache_resource
def load_sms_classifier():
    try:
        return pipeline("text-classification", model="JosephWalter69/ghana-momo-sms-classifier")  # ← change to YOUR actual HF repo
    except Exception as e:
        st.warning("Could not load advanced SMS model from Hugging Face. Using keyword fallback.")
        return None

sms_classifier = load_sms_classifier()

# ────────────────────────────────────────────────
#   TABS
# ────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📱 SMS Check", "🔗 Link Check", "💰 Transaction Cross Check",
    "📜 History", "📖 How to Use", "ℹ️ About"
])

# ────────────────────────────────────────────────
#   TAB 1 – SMS CHECK
# ────────────────────────────────────────────────
with tab1:
    st.subheader("Paste suspicious SMS")
    sms_text = st.text_area("SMS Message", height=180, placeholder="...")

    if st.button("Check SMS", type="primary"):
        if sms_text.strip():
            if sms_classifier:
                result = sms_classifier(sms_text)[0]
                label = result['label']
                score = result['score']
                is_fake = (label == 'LABEL_1')  # adjust if your training uses different labels
                confidence = score if is_fake else (1 - score)
            else:
                fake_keywords = ["send back", "return money", "wrong transfer", "call this number", "confirm PIN"]
                is_fake = any(kw.lower() in sms_text.lower() for kw in fake_keywords)
                confidence = 0.85 if is_fake else 0.70

            if is_fake:
                st.markdown(f'<div class="alert-red">🚨 FAKE SMS! (Conf: {confidence:.0%})<br>DO NOT reply or call!</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-green">✅ Looks genuine (Conf: {confidence:.0%})</div>', unsafe_allow_html=True)

            # Familiar pattern hint (simple demo of learning)
            similar = [h for h in st.session_state.history[-10:] if h["type"] == "SMS" and h["result"] == ("FAKE" if is_fake else "GENUINE")]
            if len(similar) >= 2:
                st.caption("ℹ️ This pattern appears in recent checks")

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "type": "SMS",
                "input": sms_text[:50] + "...",
                "result": "FAKE" if is_fake else "GENUINE"
            })

            # Feedback buttons
            st.markdown("Was this correct?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes", key=f"sms_yes_{len(st.session_state.history)}"):
                    st.success("Thank you for confirming!")
                    log_feedback("SMS", sms_text, "GENUINE" if not is_fake else "FAKE", "correct")
            with col2:
                wrong_comment = st.text_input("What was wrong? (optional)", key=f"sms_wrong_{len(st.session_state.history)}")
                if st.button("👎 Wrong", key=f"sms_no_{len(st.session_state.history)}"):
                    correct_label = "GENUINE" if is_fake else "FAKE"
                    st.info(f"Marked as {correct_label}. Thanks!")
                    log_feedback("SMS", sms_text, correct_label, "wrong", wrong_comment)
        else:
            st.warning("Please enter an SMS first.")

# ────────────────────────────────────────────────
#   TAB 2 – LINK CHECK (feedback added)
# ────────────────────────────────────────────────
with tab2:
    st.subheader("Paste suspicious link")
    link_url = st.text_input("Link/URL", placeholder="https://...")

    if st.button("Verify Link", type="primary"):
        if link_url.strip():
            official = ["mtn.com.gh", "momo.mtn.com", "telecel", "airteltigo"]
            suspicious = ["login", "verify", "pin", "claim", "refund", ".tk", ".ml", ".xyz"]
            domain = link_url.lower().replace("https://","").split("/")[0]
            is_phish = not any(d in domain for d in official) or any(k in link_url.lower() for k in suspicious)

            if is_phish:
                st.markdown('<div class="alert-red">🚨 PHISHING LINK!<br>DO NOT CLICK!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-green">✅ Safe link</div>', unsafe_allow_html=True)

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "type": "Link",
                "input": domain,
                "result": "PHISHING" if is_phish else "SAFE"
            })

            # Feedback
            st.markdown("Was this correct?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes", key=f"link_yes_{len(st.session_state.history)}"):
                    st.success("Thank you!")
                    log_feedback("Link", link_url, "SAFE" if not is_phish else "PHISHING", "correct")
            with col2:
                wrong_comment = st.text_input("What was wrong?", key=f"link_wrong_{len(st.session_state.history)}")
                if st.button("👎 Wrong", key=f"link_no_{len(st.session_state.history)}"):
                    correct_label = "SAFE" if is_phish else "PHISHING"
                    st.info(f"Marked as {correct_label}")
                    log_feedback("Link", link_url, correct_label, "wrong", wrong_comment)
        else:
            st.warning("Please enter a link.")

# ────────────────────────────────────────────────
#   TAB 3 – TRANSACTION CHECK (feedback added)
# ────────────────────────────────────────────────
with tab3:
    st.subheader("Enter transaction details")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (GHS)", min_value=0.0, value=500.0, step=10.0)
        old_bal = st.number_input("Old Balance", min_value=0.0, value=2000.0, step=100.0)
    with col2:
        new_bal = st.number_input("New Balance", min_value=0.0, value=1500.0, step=100.0)
        trans_type = st.selectbox("Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

    if st.button("Check Transaction", type="primary"):
        # Your existing transaction logic here (IsolationForest etc.)
        # ... (keep your current code for model & prediction) ...

        # After result display (is_fraud variable):
        if is_fraud:
            st.markdown('<div class="alert-red">🚨 SUSPICIOUS TRANSACTION!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-green">✅ Looks normal</div>', unsafe_allow_html=True)

        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M"),
            "type": "Transaction",
            "input": f"{trans_type} GHS {amount}",
            "result": "FRAUD" if is_fraud else "GENUINE"
        })

        # Feedback
        st.markdown("Was this correct?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes", key=f"tx_yes_{len(st.session_state.history)}"):
                st.success("Thank you!")
                log_feedback("Transaction", f"{trans_type} {amount}", "GENUINE" if not is_fraud else "FRAUD", "correct")
        with col2:
            wrong_comment = st.text_input("What was wrong?", key=f"tx_wrong_{len(st.session_state.history)}")
            if st.button("👎 Wrong", key=f"tx_no_{len(st.session_state.history)}"):
                correct_label = "GENUINE" if is_fraud else "FRAUD"
                st.info(f"Marked as {correct_label}")
                log_feedback("Transaction", f"{trans_type} {amount}", correct_label, "wrong", wrong_comment)

# ────────────────────────────────────────────────
#   TAB 4 – HISTORY
# ────────────────────────────────────────────────
with tab4:
    st.subheader("Check History")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No checks yet.")

# ────────────────────────────────────────────────
#   TAB 5 & 6 – unchanged (How to Use & About)
# ────────────────────────────────────────────────
# (keep your existing code for these tabs)

# ────────────────────────────────────────────────
#   SIDEBAR – feedback summary
# ────────────────────────────────────────────────
with st.sidebar:
    st.success("System running")
    if os.path.exists(FEEDBACK_FILE):
        try:
            df_fb = pd.read_csv(FEEDBACK_FILE)
            total = len(df_fb)
            wrongs = len(df_fb[df_fb['user_judgment'] == 'wrong'])
            st.markdown(f"**Feedback received**: {total} checks")
            if wrongs > 0:
                st.caption(f"{wrongs} corrections logged – improving over time")
        except:
            st.caption("Feedback summary loading...")

st.caption("MoMo Fraud Guard – Final Year Project")
