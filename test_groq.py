from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

load_dotenv()

key = os.getenv("GROQ_API_KEY")
if not key:
    print("❌ API key not found — check your .env file")
else:
    print(f"✅ API key found: {key[:8]}...")
    
    llm = ChatGroq(
        groq_api_key=key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    
    response = llm.invoke([HumanMessage(content="Say hello in one sentence.")])
    print(f"✅ Groq connected successfully!")
    print(f"Response: {response.content}")