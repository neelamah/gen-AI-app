# https://github.com/campusx-official/langgraph-tutorials/blob/main/7_review_reply_workflow.ipynb

#RAG
 #RAG is a technique where a language model (like GPT) generates responses by combining its own generative capabilities with information retrieved from external sources, such as documents, databases, or knowledge bases.
 
 #Key points:
#1.Retrieval: The system searches for relevant information from a vector store, database, or documents based on the user query.
#2.Augmentation: The retrieved information is provided as context to the language model.
#3.Generation: The language model generates a response that uses both its internal knowledge and the retrieved context.

#In LangChain terms:
#1.You typically use a Retriever (like FAISS or Chroma) to fetch relevant documents.
#2.Then pass those documents to a LLMChain or RAGChain to generate an informed response.

#User Query → Retriever fetches relevant docs → pass (context(retrived by retriver) + query) to LLM generates response using docs → Response to user


#Why Rag:-
#1. LLM outdated knowledge
#2. privay:- data remain private.
#3. Hallucinations:- By providing relevant retrieved context, RAG reduces the chance the LLM will make up answers.
#Hallucinations occur when a language model generates information that is false, misleading, or not supported by the retrieved documents or real-world facts.


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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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

#load pdf
loader = PyPDFLoader("intro-to-ml.pdf")
docs = loader.load()

#split document using spliter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)


# create embadding
embeddings = HuggingFaceEmbeddings(model='text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)

#create retriver
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

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

4. #Rag tool
@tool
def rag_tool(query):

    """
    Retrieve relevant information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query': query,
        'context': context,
        'metadata': metadata
    }

#---------------------------End tool-----------------

#----------------Register tool with LLM---------------
#create list of tool want to register
tools = [search_tool, get_stock_price, calculator, rag_tool]
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
    
    