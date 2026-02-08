# https://github.com/campusx-official/langgraph-tutorials/blob/main/7_review_reply_workflow.ipynb

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages




# -------------------------
# LLM SETUP
# -------------------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion",
    huggingfacehub_api_token="", # add your hugging face token here.
    max_new_tokens= 10
)

# create model
model = ChatHuggingFace(llm=llm)



# -------------------------
# STATE
# -------------------------
class ChatState(TypedDict):
    # add_message ( more optimise with basemessage instead of add method for reducer.)
    messages: Annotated[list[BaseMessage], add_messages]    # BaseMessage :- all message(AI message, humMessage, System meessage, tool message) inherit form base Message. so In list any message can save.

# -------------------------
# NODES
# -------------------------


#function chat_board
def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']
    # send to llm
    response = model.invoke(messages)
    # response store state
    return {'messages': [response]}

# create StaeGraph object
graph = StateGraph(ChatState)

# create node
graph.add_node('chat_node', chat_node)  #chat_node is a  python function. for node, which run when node execute. and state is ChatState which is input and output of node. and in node we take messages from state and send to model and get response and return in state.

# create edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile graph
chatbot = graph.compile()

# execute graph
initial_state = {
    'messages': [HumanMessage(content='What is the capital of india')]
}

print("Initial State:", initial_state)  # Debugging line

result = chatbot.invoke(initial_state)['messages'][-1].content
print("Chatbot Response:", result)

# while True:
#     user_message = input("type here")
    
    
#     if user_message.strip().lower() in ['exit' 'quit' 'bye']:
#         break
#     response = chatbot.invoke({'messages': HumanMessage(content=user_message)})
#     print(response['messages'][-1].content)

