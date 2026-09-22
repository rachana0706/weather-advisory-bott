import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import graph
from dotenv import load_dotenv

load_dotenv()

st.title("Weather-Advisory Support Bot")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Ask about outdoor activity safety...")

if user_input:
    st.session_state["messages"].append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state["session_id"]}}
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            inputs = {"messages": [HumanMessage(content=user_input)]}
            output_state = graph.invoke(inputs, config=config)
            
            final_msg = output_state["messages"][-1]
            st.markdown(final_msg.content)
            
            st.session_state["messages"] = output_state["messages"]
