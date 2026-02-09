import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage

#chat interface using streamlit
#Add streaming response from chatbot to streamlit frontend.(it wil not for complete response , it will start streaming when start getting tocken from LLM. and it will update response in frontend when get new token from LLM. so user can see response in real time without waiting for complete response from LLM.)
#Added sidebar for conversation history and clear conversation button.
#thread id will be dynamic.
#when we write in input box and hit enter , stramlit code start from starting due to which we will lose our conversation history. to avoid this we need to use session state of streamlit. and we will store our conversation history in session state and when we get new response from LLM we will update our session state and display in sidebar.
#st.session_state is a dictionary-like object , on enter press it does not loss history. when we refersh manually then it will loss history, In that casse we can use database.

# Initialize conversation history in session state if it doesn't exist
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []  # Initialize conversation history in session state
    
#it will read conversation history from session state and display chat messages.
#we are doing this because when we press enter it will start from starting and we will lose our conversation history. so to avoid this we need to store our conversation history in session state and when we get new response from LLM we will update our session state and display chat
for message in st.session_state.conversation_history:
    with st.chat_message(message['role']):
        st.text(message['content'])
    
user_input = st.chat_input('Type here...')

#Create thread id for converstion for persistance storage.
config = {'configurable': {'thread_id': '1'}} 

# display user input and assistant response in chat format
if user_input:
    # Add user message to conversation history
    st.session_state.conversation_history.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    #invoke LLM and get response
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]},config=config)
    ai_message = response['messages'][-1].content
    
    # Add assistant response to conversation history
    #without streaming response from chatbot to streamlit frontend.
    # st.session_state.conversation_history.append({'role': 'assistant', 'content': ai_message })  # Initialize assistant message in conversation history
    # with st.chat_message("assistant"):
    #       st.text(ai_message)
    
    
    # Apply streaming response from chatbot to streamlit frontend.
    # output of chatbot.stream having message_chunk, metadata , it is mention in steamlit docs, so we will filter content
    ai_message= st.write_stream(
        message_chunk.content for message_chunk, metadata in chatbot.stream({'messages': [HumanMessage(content=user_input)]}, config=config, stream_mode='messages')
    )
    st.session_state.conversation_history.append({'role': 'assistant', 'content': ai_message })  # Update assistant message in conversation history
    

    

    

