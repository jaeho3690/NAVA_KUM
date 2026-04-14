from __future__ import annotations

import hmac
import os

import streamlit as st


AUTH_SESSION_KEY = "doctor_labeling_authenticated"
DEFAULT_SHARED_PASSWORD = "navalabel"


def _expected_password() -> str:
    return os.environ.get("NAVA_DOCTOR_LABEL_PASSWORD", DEFAULT_SHARED_PASSWORD)


def require_shared_password(app_label: str) -> None:
    if st.session_state.get(AUTH_SESSION_KEY, False):
        with st.sidebar:
            st.markdown("---")
            st.caption("Shared access: logged in")
            if st.button("Logout", key="doctor_app_logout", width="stretch"):
                st.session_state[AUTH_SESSION_KEY] = False
                st.rerun()
        return

    st.title(app_label)
    st.markdown(
        "<h3 style='color:#b45309; margin-bottom:0.4rem;'>공용 비밀번호를 입력해야 접근할 수 있습니다.</h3>",
        unsafe_allow_html=True,
    )
    st.info("접근 권한이 있는 사용자만 공용 비밀번호로 로그인하세요.")

    with st.form("doctor_app_login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary", width="stretch")

    if submitted:
        if hmac.compare_digest(password, _expected_password()):
            st.session_state[AUTH_SESSION_KEY] = True
            st.rerun()
        st.error("비밀번호가 올바르지 않습니다.")

    st.stop()
