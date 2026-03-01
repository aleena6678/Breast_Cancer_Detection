import streamlit as st
import json
import bcrypt
import os
import time
import re
from datetime import datetime

USERS_FILE = "doctor_users.json"

# -----------------------------
# User Management Functions
# -----------------------------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return {}

def verify_login(username, password):
    users = load_users()
    if username not in users:
        return False, None
    
    stored_hash = users[username]["password"]
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode('utf-8')
        
    try:
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return True, users[username]["name"]
    except ValueError:
        pass
        
    return False, None

def register_user(username, password, doctor_id):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = {
        "password": hashed_password,
        "name": doctor_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)
        
    return True, "Registration successful!"

def clear_login_fields():
    if "user_input_login" in st.session_state:
        st.session_state["user_input_login"] = ""
    if "pass_input_login" in st.session_state:
        st.session_state["pass_input_login"] = ""

# -----------------------------
# CSS Styling
# -----------------------------
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #f8fafc;
        background-attachment: fixed;
    }
    
    .professional-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    .main-title {
        font-size: 32px;
        font-weight: 700;
        text-align: center;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 15px;
        margin-bottom: 30px;
    }
    
    .form-label {
        color: #334155;
        font-weight: 500;
        font-size: 14px;
        margin-bottom: 6px;
        display: block;
    }
    
    .stTextInput > div > div > input {
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
        padding: 10px 14px !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }
    
    .stButton > button {
        border-radius: 6px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0 !important;
        padding: 12px 20px !important;
        background: transparent !important;
        border: none !important;
        color: #64748b;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Dialogs
# -----------------------------
@st.dialog("Authorize Registration")
def auth_and_register_dialog(new_username, new_password, new_doctor_id):
    st.markdown("Please enter the credentials of an **already registered doctor** to authorize this new account.")
    
    with st.form("auth_form", clear_on_submit=False):
        st.markdown('<div class="form-label">🔐 Authorizing Username</div>', unsafe_allow_html=True)
        auth_username = st.text_input("", key="auth_uname", label_visibility="collapsed")
        
        st.markdown('<div class="form-label">🔑 Authorizing Password</div>', unsafe_allow_html=True)
        auth_password = st.text_input("", type="password", key="auth_pass", label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        auth_submit = st.form_submit_button("Authorize & Create Account", type="primary", use_container_width=True)
        
        if auth_submit:
            if not auth_username or not auth_password:
                st.error("⚠️ Please enter authorizing credentials.")
            else:
                with st.spinner("Authorizing..."):
                    time.sleep(0.5)
                    ok, _ = verify_login(auth_username, auth_password)
                    if ok:
                        success, msg = register_user(new_username, new_password, new_doctor_id)
                        if success:
                            st.success(f"✅ {msg}")
                            st.info("Registration complete. Redirecting...")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                    else:
                        st.error("❌ Unauthorized: Invalid doctor credentials.")

# -----------------------------
# Main Authentication UI
# -----------------------------
def doctor_login():
    load_css()
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "doctor_name" not in st.session_state:
        st.session_state.doctor_name = None

    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        
        with col2:
            st.markdown("""
                <div class="professional-card">
                    <div style="text-align: center; font-size: 60px; margin-bottom: 10px;">🏥</div>
                    <div class="main-title">Medical Portal</div>
                    <div class="subtitle">Secure Breast Cancer Detection System</div>
            """, unsafe_allow_html=True)
            
            # The tabs and the login form container
            tab_login, tab_register = st.tabs(["🔐 Login", "📝 Add new doctor"])
            
            # LOGIN TAB
            with tab_login:
                with st.form("login_form", clear_on_submit=False):
                    st.markdown('<div class="form-label">👤 Username</div>', unsafe_allow_html=True)
                    username = st.text_input("", placeholder="Enter your username", key="user_input_login", label_visibility="collapsed")
                    
                    st.markdown('<div class="form-label">🔒 Password</div>', unsafe_allow_html=True)
                    password = st.text_input("", placeholder="Enter your password", type="password", key="pass_input_login", label_visibility="collapsed")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        submit = st.form_submit_button("Log In", type="primary", use_container_width=True)
                    with col_b:
                        clear = st.form_submit_button("Clear", on_click=clear_login_fields, use_container_width=True)
                    
                    if submit:
                        if not username or not password:
                            st.error("⚠️ Please enter both username and password.")
                        else:
                            with st.spinner("Authenticating..."):
                                time.sleep(0.5)
                                ok, doctor_name = verify_login(username, password)
                                if ok:
                                    st.session_state.logged_in = True
                                    st.session_state.doctor_name = doctor_name
                                    st.success(f"✅ Welcome back, Dr. {doctor_name}")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("❌ Invalid credentials. Please try again.")

            # REGISTER TAB
            with tab_register:
                with st.form("register_form", clear_on_submit=True):
                    st.markdown('<div class="form-label">👨‍⚕️ Doctor ID</div>', unsafe_allow_html=True)
                    new_doctor_id = st.text_input("", placeholder="e.g. DOC-12345", label_visibility="collapsed")
                    
                    st.markdown('<div class="form-label">👤 Username</div>', unsafe_allow_html=True)
                    new_username = st.text_input("", placeholder="Choose a unique username", label_visibility="collapsed")
                    
                    st.markdown('<div class="form-label">🔒 Password</div>', unsafe_allow_html=True)
                    new_password = st.text_input("", placeholder="Create a strong password", type="password", label_visibility="collapsed")
                    
                    st.markdown('<div class="form-label">🔒 Confirm Password</div>', unsafe_allow_html=True)
                    confirm_password = st.text_input("", placeholder="Repeat the password", type="password", label_visibility="collapsed")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    reg_submit = st.form_submit_button("Register Account", type="primary", use_container_width=True)
                    
                    if reg_submit:
                        if not new_doctor_id or not new_username or not new_password:
                            st.error("⚠️ Please fill out all fields.")
                        elif not re.match(r"^DOC-\d{5}$", new_doctor_id):
                            st.error("⚠️ Doctor ID must be in the format 'DOC-12345'.")
                        elif new_password != confirm_password:
                            st.error("⚠️ Passwords do not match.")
                        elif len(new_password) < 6:
                            st.error("⚠️ Password must be at least 6 characters long.")
                        else:
                            auth_and_register_dialog(new_username, new_password, new_doctor_id)
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    doctor_login()