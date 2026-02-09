# https://github.com/campusx-official/langgraph-tutorials/blob/main/7_review_reply_workflow.ipynb

#Tools
# In LangGraph, tools are external functions or APIs that an LLM can call to perform actions like calculations, data access, or API calls, enabling the agent to act beyond text generation.

#1. tool_node :- “ToolNode in LangGraph is a graph node responsible for executing tools when the LLM requests a tool call and returning the tool output back into the graph state.”
#2. tools_condition :- “tools_condition is a routing condition in LangGraph that checks whether the LLM response contains a tool call and directs execution to the ToolNode if required.”
#3. Register tools:-  Need to  register tool with LLM.

#Benifit of tools:-
#Tools let your agent:
#1.Call APIs
#2.Query databases
#3.Run Python logic
#4.Search the web
#5.Perform calculations
#6.Interact with files

#Some common tools:--
#1.search_tool → search the web
#2.calculator → math
#3.sql_tool → database queries
#4.file_tool → read/write files
#5.api_tool → REST APIs
#6.python_tool → data processing

#steps to follow to use tools:-
#1. Define the tool:- Create a callable function (custom or LangChain tool) with a clear schema.
    #Register the tool with the LLM Bind the tool so the LLM knows when and how to call it.
#2. Create a ToolNode:- Add the tool(s) to a ToolNode for execution inside the graph.
#3.Design the graph flow:-  Connect the LLM node and ToolNode using edges and conditions. 
#4. Enable tool invocation: Let the LLM decide when to call a tool based on user input.
     #( Note:- LLm will return tool Name , which tool need to execute, will not execute tool here.)

#5. Execute the tool:-  LangGraph routes the request to the ToolNode and runs the tool.
#6. Return tool output to the LLM:- The result is added to the graph state.
#7. Generate the final response: The LLM uses the tool output to respond or continue the workflow.


#One-liner (interview version)

#“To use tools in LangGraph, we define and register tools, add them to a ToolNode, connect them in the graph, and let the LLM invoke them during execution.”



from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun 
from langchain_core.tools import tool
import requests


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


#-------------------------- tools---------------------

#1. search tool
search_tool = DuckDuckGoSearchRun(region="us-en")
print(search_tool.invoke("latest AI news"))

#2. Custom tool (calculation)
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
#3. custom  API call tool
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

#---------------------------End tool-----------------

#----------------Register tool with LLM---------------
#create list of tool want to register
tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)  #  In higging face bind method not working.


#-------------------------------------------------

#---------------------- Node function-----------------
#function chat_board
def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']
    # send to llm
    response = model.invoke(messages)
    # response store state
    return {'messages': [response]}

#created tool Node 
tool_node = ToolNode(tools)



#-------------------------------------------------------------------------

# create StateGraph object and pass state to it.
#Now our graph will have complete state of conversation.
graph = StateGraph(ChatState)

# create node
graph.add_node('chat_node', chat_node)  #chat_node is a  python function. for node, which run when node execute. and state is ChatState which is input and output of node. and in node we take messages from state and send to model and get response and return in state.
graph.add_node("tools", tool_node)

# create edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

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


# DuckDuckGo
print(search_tool.invoke("latest AI news"))

# Calculator
print(calculator(first_num=5, second_num=3, operation="mul"))

# Stock API
print(get_stock_price("AAPL"))


#excute simple flow to check in backend.
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


#apply streaming in llm respponse to check in backaend
# for message_chunk, metadata in chatbot.stream(initial_state, config=config, stream_mode='messages'):
#     if message_chunk.content:
#         print(message_chunk.content, end='', flush=True)  # Print the content as it arrives without adding a new line
    
    