"""Run with: python -m streamlit run app.py"""
import logging
import time

import streamlit as st

from src.agent_pipeline import SupportAgentPipeline

st.set_page_config(page_title="Apple Help", page_icon="🍎", layout="centered")


@st.cache_resource(show_spinner="Preparing the support library…")
def get_pipeline():
    # Shared read-only index/client; conversation stays in each browser session.
    return SupportAgentPipeline()


st.title("🍎 Apple Help")
st.caption("An independent AI assistant for your Apple questions.")
with st.sidebar:
    st.header("Your conversation")
    st.write("Ask about your iPhone, Mac, iCloud, or other Apple services. Include your device and software version if you know them.")
    st.caption("Answers use historical support examples, which may be outdated. This app cannot access your account or contact Apple for you.")
    st.caption("Questions and recent chat context are sent to the configured AI provider. Do not share passwords or verification codes.")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_answer(message):
    st.markdown(message["content"])
    if message.get("escalated"):
        st.info("For this issue, please contact Apple Support directly at https://support.apple.com/.")
    if "seconds" in message:
        st.caption(f"Answered in {message['seconds']:.1f}s")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        show_answer(message)

if not st.session_state.messages:
    st.write("What can I help you with?")
    st.caption('Try: “My iPhone battery drains quickly after an update.”')

if question := st.chat_input("Ask an Apple question…", max_chars=4000):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            started = time.perf_counter()
            with st.spinner("Finding an answer…"):
                result = get_pipeline().process_query(question, history=st.session_state.messages)
            answer = {"role": "assistant", "content": result["drafted_reply"],
                      "escalated": result["action"] == "ESCALATE_TO_HUMAN",
                      "seconds": time.perf_counter() - started}
            st.session_state.messages.extend([
                {"role": "user", "content": question}, answer])
            show_answer(answer)
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
        except Exception:
            logging.exception("Support response failed")
            st.error("I couldn't finish that answer. Please send your question again. If this continues, check the API configuration and connection.")
