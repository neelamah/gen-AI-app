# https://github.com/campusx-official/langgraph-tutorials/blob/main/7_review_reply_workflow.ipynb

#HITL
#Human-in-the-Loop (HITL) is a process where humans interact with AI/ML systems to improve accuracy, guide decisions, or validate outputs.

#Purpose:-

#1.Improves Accuracy – Humans correct AI errors, making outputs more reliable.
#2.Reduces Bias – Human oversight helps identify and fix biased predictions.
#3.Handles Complex Cases – AI struggles with ambiguity; humans make judgement calls.
#4.Enables Continuous Learning – Human feedback can retrain and fine-tune the AI over time.

#eg:- before payment need huma approval()

#in this applied only in backend.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun 
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
import requests


# -------------------------
# LLM SETUP
# -------------------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
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


#-------------------------- tools---------------------
def chat_node(state: ChatState):

    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a user question.",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? yes/no"
    })
    
    if decision["approved"] == 'no':
        return {"messages": [AIMessage(content="Not approved.")]}

    else:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}


#-------------------------------------------------------------------------


graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)



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
#excute simple flow to check in backend.
result = chatbot.invoke(initial_state, config=config)

print(result)

#check intrupt message
message = result['__interrupt__'][0].value
print(message)


#sending msg to take spproval from user
user_input = input(f"\nBackend message - {message} \n Approve this question? (y/n): ")


# Resume the graph with the approval decision
final_result = chatbot.invoke(
    Command(resume={"approved": user_input}),
    config=config,
)

#fimal result
print(final_result["messages"][-1].content)

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


