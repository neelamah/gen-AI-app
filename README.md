# chat-bot-genAI
Chatbot using Gen AI


1️⃣ Project Architecture (Full GenAI)

                  ┌─────────────┐
                  │   User UI   │  ← Streamlit Chat
                  └─────┬───────┘
                        │
                        ▼
                ┌─────────────┐
                │ LangGraph   │  ← Orchestration
                │ (Nodes +    │
                │  Conditional│
                │  Flow)      │
                └─────┬───────┘
      ┌───────────────┼─────────────────┐
      ▼               ▼                 ▼
┌─────────┐     ┌─────────┐       ┌──────────┐
│ Memory  │     │ Tool    │       │ Prompt   │
│ Node    │     │ Node    │       │ Templates│
└─────────┘     └─────────┘       └──────────┘
      │               │                 │
      └─────┬─────────┴───────────────┘
            ▼
     ┌─────────────┐
     │ TinyLlama   │  ← Local LLM
     └─────┬───────┘
           ▼
     ┌─────────────┐
     │ Output      │  ← Output Parser / Post-processing
     │ Parser      │
     └─────────────┘
           ▼
        Response


Key concepts implemented:

1.Memory – stores conversation history, optionally summarizes long chats
2.Tool nodes – calculator, search, API calls
3.Conditional nodes – sequential, branching, parallel flows
4.Prompt templates – system + user instructions
5.Streaming responses – token-by-token generation
6.Output parsers – structured output (JSON, QA, etc.)
7.RAG – Retrieval-Augmented Generation, plug in knowledge sources

2️⃣ Install Required Libraries

1.faiss-cpu → for local vector database for RAG
2.langgraph → orchestration
3.transformers + torch → TinyLlama LLM modal
4.streamlit → UI
5. Python 3.11 (important!)

how to install:- 
%pip install streamlit
#%pip install langgraph
#%pip install langchain
#%pip install langchain-huggingface
#%pip install transformers
#pip install torch
#%pip install sentencepiece faiss-cpu

Agentics AI project:-
AI Trip Planner Chatbot — Ready-to-Build Structure
Project Goal
    1.User enters a trip request (e.g., “Plan a 3-day trip to Gurgaon”).
    2.AI chatbot breaks the request into steps: weather check, sightseeing, hotels, daily plan.
    3.emo-ready: Interactive chat interface showing agentic reasoning.