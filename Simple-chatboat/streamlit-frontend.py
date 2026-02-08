import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage


user_input = st.chat_input('Type here...')

# display user input and assistant response in chat format
if user_input:
    with st.chat_message("user"):
        st.text(user_input)
        
    with st.chat_message("assistant"):
        # response = chatbot.invoke(input={'messages': [HumanMessage(content=user_input)]}, stream=True)
          response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]})
          
          ai_message = response['messages'][-1].content
          st.text(ai_message)
          

    

    

