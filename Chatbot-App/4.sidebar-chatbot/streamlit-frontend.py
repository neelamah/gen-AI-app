import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import uuid



#chat interface using streamlit
#Add streaming response from chatbot to streamlit frontend.(it wil not for complete response , it will start streaming when start getting tocken from LLM. and it will update response in frontend when get new token from LLM. so user can see response in real time without waiting for complete response from LLM.)
#Added sidebar for conversation history and clear conversation button.
#thread id will be dynamic.
#when we write in input box and hit enter , stramlit code start from starting due to which we will lose our conversation history. to avoid this we need to use session state of streamlit. and we will store our conversation history in session state and when we get new response from LLM we will update our session state and display in sidebar.
#st.session_state is a dictionary-like object , on enter press it does not loss history. when we refersh manually then it will loss history, In that casse we can use database.



#IMPOrtant
#we are preparing  message_history, thread_id, chat_threads(this list will have all thread to display in sidebar)
# we are storing evering this session storage and to display reading from session storage.

#e use session storage because when we end msg streamlit start from stating an dit empy all message . so we use session storage.

# **************************************** utility functions *************************
# Generate a unique thread ID for each conversation
def generate_thread_id():
    thread_id = uuid.uuid4() 
    return thread_id

#on new chat button click 3 thins to de:-
#1. create new thread id 
#2. save new thread id in session state
#3. clear conversation history in session state(we don't to any thing when we start new chat):- call  reset_chat() where we have created new chat button.
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

# create a list which have all threads of chat which we can read and show in sidebar
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}}) # it will filter chat history of that thread id and return only that conversation history.
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
# Initialize chat history in session state if it doesn't exist
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:   #[::-1] is used to reverse the list so that latest conversation will be on top in sidebar.
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

# we are doing here because  load_conversation function return list of messages in format of {'role': 'user/assistant', 'content': 'message content'} but our chatbot backend is using HumanMessage and AIMessage format so we need to convert it to that format.
#then we will update our session state with that messages so that it will display msg using code write for msg display.
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

#it will read conversation history from session state and display chat messages.
#we are doing this because when we press enter it will start from starting and we will lose our conversation history. so to avoid this we need to store our conversation history in session state and when we get new response from LLM we will update our session state and display chat
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

     # first add the message to message_history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})