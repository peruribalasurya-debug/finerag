import sys
import os
sys.path.append(".")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from retrieval.vectorstore import load_vectorstore, search

load_dotenv()

def build_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

def format_context(results: list) -> str:
    context = ""
    for i, r in enumerate(results):
        context += f"\n[Source {i+1} - Page {r['metadata']['page']}]\n"
        context += r["content"]
        context += "\n"
    return context

def ask(query: str, top_k: int = 10) -> dict:
    index, chunks = load_vectorstore()

    # Semantic search
    results = search(query, index, chunks, top_k=top_k)
    seen_contents = set(r["content"] for r in results)

    # Keyword boost — find chunks with strong keyword matches
    query_words = [w for w in query.lower().split() if len(w) > 3]
    keyword_results = []

    for chunk in chunks:
        content_lower = chunk["page_content"].lower()
        matches = sum(1 for kw in query_words if kw in content_lower)
        if matches >= 2 and chunk["page_content"] not in seen_contents:
            keyword_results.append({
                "content": chunk["page_content"],
                "metadata": chunk["metadata"],
                "score": 0.0,
                "keyword_matches": matches
            })
            seen_contents.add(chunk["page_content"])

    # Sort keyword results by match count and take top 5
    keyword_results.sort(key=lambda x: x["keyword_matches"], reverse=True)
    combined = results + keyword_results[:5]
    context = format_context(combined)

    llm = build_llm()

    messages = [
        SystemMessage(content="""You are a financial analyst assistant specialized
        in analyzing SEC filings and annual reports.
        Answer questions using ONLY the provided context.
        If the answer is not in the context, say 'I could not find this information in the document.'
        Always mention the page number where you found the information.
        Be precise with numbers and financial figures.
        When mentioning dollar amounts, write them as 'USD 383 million' or '383 million dollars' instead of using the dollar sign."""),
        HumanMessage(content=f"""Context from Apple's 10-K filing:
{context}

Question: {query}

Answer:""")
    ]

    response = llm.invoke(messages)

    return {
        "question": query,
        "answer": response.content,
        "sources": combined
    }


if __name__ == "__main__":
    questions = [
        "What was Apple's total revenue in 2023?",
        "What were Apple's main risk factors in 2023?",
        "How much did Apple spend on research and development in 2023?"
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"Q: {question}")
        print("=" * 60)
        result = ask(question)
        print(f"A: {result['answer']}")

        source_pages = [f"Page {s['metadata']['page']}" for s in result['sources']]
        print(f"\nSources used: {source_pages}")