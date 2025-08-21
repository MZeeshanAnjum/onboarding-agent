import time
import streamlit as st
import requests

from requirements_agent.ob_agent import OnboardingAgent, AgentState, BusinessInfoChecklist
from config import FASTAPI_URL, GET_RAG_AGENT_URL
from requirements_agent.utils.rag import process_document, Initialize_vector_store

st.set_page_config(page_title="🧠 Requirements Agent", page_icon="🤖")

st.title("🧠 Requirements Agent")

# ---- Initialize session state ----
if "agent" not in st.session_state:
    st.session_state.agent = OnboardingAgent()
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState()
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False  # controls uploader visibility
if "first_run" not in st.session_state:
    st.session_state.first_run = True
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

agent = st.session_state.agent
state = st.session_state.agent_state


# ---- Document uploader trigger ----
with st.sidebar:
    if not st.session_state.show_uploader:
        if st.button("📄 Upload Document"):
            st.session_state.show_uploader = True
            st.rerun()
    st.markdown("---")
# ---- Sidebar summary ----
st.sidebar.markdown("## 📝 Summary")
if state.summary:
    st.sidebar.markdown(state.summary)
else:
    st.sidebar.markdown("*No summary available yet.*")

# ---- Conversation display ----
st.markdown("### 💬 Conversation")
last_role = None
context_history = state.context_history[1:] if state.documents_uploaded else state.context_history

filtered_history = []
skip_buffer = None

for message in context_history:
    role = message["role"]

    if role == "assistant":
        # Keep replacing until a different role comes
        skip_buffer = message
    else:
        # If buffer exists, push it before adding user message
        if skip_buffer:
            filtered_history.append(skip_buffer)
            skip_buffer = None
        filtered_history.append(message)

# If the conversation ends with assistant messages, keep the last one
if skip_buffer:
    filtered_history.append(skip_buffer)

# Now render
for message in filtered_history:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])

# ---- Chat input ----
user_input = st.chat_input("Ask something...")
# Inject default input on first run (e.g., after document upload)
# if not st.session_state.first_run and st.session_state.documents_processed:
#     user_input = "HI"  # Default question for first run
#     st.session_state.first_run = True

if user_input or (st.session_state.documents_processed and  st.session_state.first_run):
    if not user_input:
        user_input = ""
    with st.chat_message("user"):
        if user_input != "":
            st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state.query = user_input
            try:
                payload = state.model_dump() 
                print(f"Payload for agent invocation: {payload}")
                response = requests.post(FASTAPI_URL, json=payload, timeout=800)
                response.raise_for_status()
                result = response.json()

                updated_state_dict = result.get("result", {})
                data_value = updated_state_dict.get("data")

                if not isinstance(data_value, BusinessInfoChecklist):
                    if isinstance(data_value, dict):
                        data_value = BusinessInfoChecklist(**data_value)
                    else:
                        data_value = data_value.model_dump()

                updated_state_dict["data"] = data_value
                st.session_state.agent_state = AgentState(**updated_state_dict)
                st.session_state.first_run = False

                last_message = updated_state_dict.get("context_history", [])[-1]
                if last_message and last_message["role"] == "assistant":
                    st.markdown(last_message["content"])
            except requests.RequestException as e:
                st.error(f"Error invoking agent: {e}")

    st.rerun()



# ---- Document uploader section ----
if st.session_state.show_uploader:
    uploaded_files = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        state.documents_uploaded = True
        st.session_state.documents_uploaded = True
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                st.info(f"**File name:** {uploaded_file.name}")
                try:
                    process_document(uploaded_file)
                    # response = requests.post(GET_RAG_AGENT_URL, json={}, timeout=500)
                    # response.raise_for_status()
                    # print(f"RAG Agent Response: {response.json()}")
                    # state.RAG_summary = response.json().get("summary", "")
                    # state.unanswered_questions = response.json().get("unanswered_questions", [])
                    # print(f"Unanswered Questions: {state.unanswered_questions}")
                    st.success(f"✅ {uploaded_file.name} processed and stored successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")

        # Hide uploader after processing
        st.session_state.show_uploader = False
        st.session_state.documents_processed = True
        st.rerun()
