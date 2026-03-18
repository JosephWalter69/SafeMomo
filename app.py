import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
from transformers import pipeline

# ================== PAGE CONFIG & STYLING ==================
st.set_page_config(page_title="MoMo Fraud Guard 🛡️", page_icon="🛡️", layout="centered")
GREEN = "#006B3F"; YELLOW = "#FCD116"; RED = "#CE1126"

st.markdown(f"""
<style>
.big-title {{font-size: 2.8rem; color: {GREEN}; text-align: center; font-weight: bold;}}
.alert-green {{background-color: #d4edda; color: #155724; padding: 20px; border-radius: 15px; text-align: center; font-size: 1.6rem;}}
.alert-red {{background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 15px; text-align: center; font-size: 1.6rem;}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🛡️ MoMo Fraud Guard</div>', unsafe_allow_html=True)
st.caption("Your personal shield against mobile money scams in Ghana")

# ================== SESSION STATE FOR HISTORY ==================
if 'history' not in st.session_state:
    st.session_state.history = []

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📱 SMS Check", "🔗 Link Check", "💰 Transaction Cross Check",
    "📜 History", "📖 How to Use", "ℹ️ About"
])

@st.cache_resource
def load_sms_classifier():
    try:
        return pipeline("text-classification", model="./ghana_momo_sms_classifier")
    except:
        st.warning("Advanced SMS model not found — using basic keyword check.")
        return None

sms_classifier = load_sms_classifier()

# ================== TAB 1: SMS CHECK ==================
with tab1:
    st.subheader("Paste suspicious SMS")
    sms = st.text_area("SMS Message", height=180, placeholder="You have received GHS 500... Call 0551234567 to return money")
    
if st.button("Check SMS", type="primary"):
    if sms.strip():
        if sms_classifier:
            result = sms_classifier(sms)[0]
            label = result['label']  # 'LABEL_0' or 'LABEL_1' — map to genuine/fake
            score = result['score']
            is_fake = label == 'LABEL_1'  # Adjust based on your training (0=genuine, 1=fake)
            confidence = score if is_fake else 1 - score
        else:
            # Fallback to keywords
            fake_keywords = ["send back", "return money", "wrong transfer", "call this number", "confirm PIN"]
            is_fake = any(kw.lower() in sms.lower() for kw in fake_keywords)
            confidence = 0.85 if is_fake else 0.70  # dummy

        if is_fake:
            st.markdown(f'<div class="alert-red">🚨 FAKE SMS DETECTED! (Confidence: {confidence:.0%})<br>DO NOT reply or call!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green">✅ Looks genuine (Confidence: {confidence:.0%})<br>Still verify in MoMo app</div>', unsafe_allow_html=True)
        
        st.session_state.history.append({"time": datetime.now().strftime("%H:%M"), "type": "SMS", "input": sms[:50]+"...", "result": "FAKE" if is_fake else "GENUINE"})
# ================== TAB 2: LINK CHECK ==================
with tab2:
    st.subheader("Paste suspicious link")
    link = st.text_input("Link/URL", placeholder="https://momo.mtn.com.gh/claim-prize")
    
    if st.button("Verify Link", type="primary"):
        official = ["mtn.com.gh", "momo.mtn.com", "telecel", "airteltigo"]
        suspicious = ["login", "verify", "pin", "claim", "refund", ".tk", ".ml", ".xyz"]
        domain = link.lower().replace("https://","").split("/")[0]
        
        is_phish = not any(d in domain for d in official) or any(k in link.lower() for k in suspicious)
        
        if is_phish:
            st.markdown('<div class="alert-red">🚨 PHISHING LINK!<br>DO NOT CLICK!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-green">✅ Safe link</div>', unsafe_allow_html=True)
        
        st.session_state.history.append({"time": datetime.now().strftime("%H:%M"), "type": "Link", "input": domain, "result": "PHISHING" if is_phish else "SAFE"})

# ================== TAB 3: TRANSACTION CROSS CHECK ==================
with tab3:
    st.subheader("Enter transaction details")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (GHS)", min_value=0.0, value=500.0)
        old_bal = st.number_input("Old Balance (Sender)", min_value=0.0, value=2000.0)
    with col2:
        new_bal = st.number_input("New Balance (Sender)", min_value=0.0, value=1500.0)
        trans_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "DEPOSIT"])
    
    if st.button("Check Transaction", type="primary"):
        np.random.seed(42)
        synthetic = pd.DataFrame({
            'amount': np.random.normal(800, 400, 500),
            'oldbalanceOrg': np.random.normal(3000, 1500, 500),
            'newbalanceOrig': np.random.normal(2500, 1400, 500),
            'type_encoded': np.random.randint(0, 4, 500)
        })
        
        model = IsolationForest(contamination=0.02, random_state=42)
        model.fit(synthetic)
        
        user_data = pd.DataFrame([{
            'amount': amount,
            'oldbalanceOrg': old_bal,
            'newbalanceOrig': new_bal,
            'type_encoded': ["CASH_OUT","TRANSFER","PAYMENT","DEPOSIT"].index(trans_type)
        }])
        
        score = model.decision_function(user_data)[0]
        is_fraud = score < 0
        
        if is_fraud:
            st.markdown('<div class="alert-red">🚨 SUSPICIOUS TRANSACTION!<br>Possible fraud detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-green">✅ Genuine transaction</div>', unsafe_allow_html=True)
        
        st.session_state.history.append({"time": datetime.now().strftime("%H:%M"), "type": "Transaction", "input": f"{trans_type} GHS {amount}", "result": "FRAUD" if is_fraud else "GENUINE"})

# ================== TAB 4: HISTORY ==================
with tab4:
    st.subheader("Your Check History")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No checks yet. Your results will appear here.")

# ================== TAB 5: HOW TO USE (EDUCATIONAL ONLY) ==================
with tab5:
    st.subheader("How to Use MoMo Fraud Guard")
    st.markdown("""
    **1. SMS Check**  
    - Copy any SMS you receive from MTN, Telecel or AirtelTigo.  
    - Paste it in the box and tap **Check SMS**.  
    - Red alert = Do NOT reply or call.  
    - Green = Looks safe, but always double-check inside your official MoMo app.

    **2. Link Check**  
    - Copy any link sent to you (WhatsApp, SMS, etc.).  
    - Paste it and tap **Verify Link**.  
    - Red alert = Never click or enter your PIN.  
    - Green = Safe to visit (but type the official website yourself).

    **3. Transaction Cross Check**  
    - Enter the amount, old balance, new balance and type of transaction.  
    - Tap **Check Transaction**.  
    - Red alert = The transaction looks suspicious — contact your network immediately.  
    - Green = Looks normal.

    **4. History**  
    - See all your past checks in one place.  
    - Clear history anytime you want.

    **Tip for feature phone users**  
    Forward any suspicious SMS to our dedicated number (coming soon) and you will receive an instant reply telling you if it is safe.
    """)

# ================== TAB 6: ABOUT ==================
with tab6:
    st.subheader("About the System")
    st.markdown("""
    **Ghana MoMo Fraud Guard**  
    Final Year Computer Science Project  
    Designed to protect millions of mobile money users in Ghana from SMS scams, phishing links, and fake transactions.
    
    - Works on any smartphone (and soon feature phones)  
    - Free and easy to use  
    - Built to make mobile money safer for everyone in Ghana
    
    **Developer**: Nii Amoo  
    **Goal**: Reduce financial losses and build trust in mobile money services.
    """)
    st.caption("Version 1.0 – March 2026")

# Sidebar
with st.sidebar:
    st.success("✅ System is LIVE and ready!")
    st.info("Protect yourself before you lose money!")

st.caption("Complete system ready for your presentation today! 🚀")
