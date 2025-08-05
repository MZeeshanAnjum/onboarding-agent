import time 
import streamlit as st
from agents_helper.onboarding_agent import OnboardingAgent, AgentState, ToolInfo

st.set_page_config(page_title="🧠 Onboarding Agent", page_icon="🤖")
st.title("🧠 Onboarding Agent Chat")

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = OnboardingAgent()

if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState()
if "selected_tools" not in st.session_state:
    st.session_state.selected_tools = []

print(st.session_state.selected_tools)

agent = st.session_state.agent
state = st.session_state.agent_state

# Show summary in the sidebar (if it exists)
st.sidebar.markdown("## 📝 Summary")

if state.summary:
    st.sidebar.markdown(state.summary)
else:
    st.sidebar.markdown("*No summary available yet.*")

if state.tools.tools_needed:
    st.sidebar.markdown("## 🛠️ Selected Tools")

    updated_selected_tools = set(st.session_state.selected_tools)  # use a set for fast lookup

    for tool in state.tools.tools_needed:
        label = tool.replace("_", " ")
        checked = tool in updated_selected_tools

        is_checked = st.sidebar.checkbox(label, value=checked, key=f"needed_tool_{tool}")

        if is_checked:
            updated_selected_tools.add(tool)
        else:
            updated_selected_tools.discard(tool)

    # Update session state only once after the loop
    st.session_state.selected_tools = list(updated_selected_tools)
    st.session_state.agent_state.tools_selected = list(updated_selected_tools)

# Display chat history in bubbles
st.markdown("### 💬 Conversation")
prev_role = None

for message in state.context_history:
    role = message["role"]
    
    # Skip system messages
    if role == "system":
        continue

    # Only display message if role is different from the previous one
    if role != prev_role:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message["content"])
        prev_role = role

if state.tools.tools_suggested:
    filtered_suggestions = [
        tool for tool in state.tools.tools_suggested if tool not in state.tools.tools_needed
    ]

    if filtered_suggestions:
        st.markdown("### 💡 Suggested Tools")

        # Divide the tools into chunks of 4
        for i in range(0, len(filtered_suggestions), 4):
            row_tools = filtered_suggestions[i:i+4]
            cols = st.columns(4)  # create 4 columns

            for col, tool in zip(cols, row_tools):
                label = tool.replace("_", " ")
                key = f"suggested_tool_{tool}"

                with col:
                    is_clicked = st.checkbox(label, key=key)
                    if is_clicked:
                        if tool not in st.session_state.selected_tools:
                            st.session_state.selected_tools.append(tool)
                        if tool not in state.tools.tools_needed:
                            state.tools.tools_needed.append(tool)
                        if tool in state.tools.tools_suggested:
                            state.tools.tools_suggested.remove(tool)
                        st.rerun()

# Prompt input
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder for assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Update query in the state
            state.query = user_input

            # Call the graph
            updated_state_dict = agent.graph.invoke(state.model_dump())

            print("Updated state dict:", updated_state_dict)

            # Convert nested Pydantic model to dict if needed
            if isinstance(updated_state_dict.get("tools"), ToolInfo):
                updated_state_dict["tools"] = updated_state_dict["tools"].model_dump()

            # Save back to session
            st.session_state.agent_state = AgentState(**updated_state_dict)
            st.session_state.selected_tools = st.session_state.agent_state.tools.tools_needed

            # Show the assistant's last message
            last_message = updated_state_dict.get("context_history", [])[-1]
            if last_message and last_message["role"] == "assistant":
                st.markdown(last_message["content"])

    st.rerun()
