# https://github.com/campusx-official/langgraph-tutorials/blob/main/7_review_reply_workflow.ipynb

# Store all conversation history. use below concept:-
#Persistent storage:-  We will use persistent storage concept to store all conversation.
#Checkpointer:- every super step become check pointer. on every check pointer it will save all values in store.
#thead Id:- it will create thread for each  conversation. (like if i start a chat and end id it , it will have a thread id , now  new chat ,it will create other thread id ofr new chat)

#Benifit of persistance storage:-
#1. short term memory:- like InMemory storage.
#2. falutl tolerance:- if any point workflow get crash, we can resume using persistnace memory.
#3. HITL(human in loop):- if we need any aproval from human , if approval take time( 1 day, 24 etc) . after getting approval we can resume state and start from that point.
#4. time travel:- we can  go back to any checkpoint point and can replay workflow again(for debugging.)

#how its work:-
#1.The graph is compiled – all nodes, state variables, and reducers are prepared to run.
#2.The checkpointer is attached to the graph(super) – it becomes the graph’s memory manager.
#3.Every state update in the graph (e.g., messages appended) is automatically saved by the checkpointer.
#4.We also use the checkpointer to load previous state using the thread_id.

#In short:
#checkpointer = tells the compiled graph how to save and load state.

#Note:- Whether it’s a super step node or regular node, any node that updates state triggers the checkpointer.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver   




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
    
    # message will look like this :- 
    # messages = [ HumanMessage("Hi"), AIMessage("Hello!")] 

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

# create StateGraph object and pass state to it.
#Now our graph will have complete state of conversation.
graph = StateGraph(ChatState)

# create node
graph.add_node('chat_node', chat_node)  #chat_node is a  python function. for node, which run when node execute. and state is ChatState which is input and output of node. and in node we take messages from state and send to model and get response and return in state.

# create edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile graph
#MemorySaver() → creates an in-memory checkpointer
#checkpointer= tells LangGraph how and where to store the graph’s state.(we have check pointer on supter steps , it will save meory.)


#create in memory checkpointer  
checkpointer = InMemorySaver()  # it is checkpointer object.
chatbot = graph.compile(checkpointer=checkpointer)  # Pass the checkpointer to the compile method.

# execute graph
initial_state = {
    'messages': [HumanMessage(content='What is the capital of india')]
}

#Create thread id for converstion.
#When you call model.invoke() again with the same thread_id, the checkpointer loads the previous state automatically. and with new query will pass to LLM.
#when ever we invoke llm need to pass thread-id.
config = {'configurable': {'thread_id': '1'}}  # You can generate a unique thread_id for each conversation.
result = chatbot.invoke(initial_state, config=config)['messages'][-1].content


# to check only on backend side its working or not. Uncomment below code. (No frontend needed)

# while True:
#     user_message = input("type here")
    
    
#     if user_message.strip().lower() in ['exit' 'quit' 'bye']:
#         break
#     response = chatbot.invoke({'messages': HumanMessage(content=user_message)},config=config)
    
#    # check final state after execution of graph.
#     print(chatbot.get_state(config))  
#     #check intermediate state history after execution of graph.( on every super node(checkpointer) what state was stored). It will show for every step.
#     print(list(chatbot.get_state_history(config)))