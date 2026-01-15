from rag.retriever import retrieve
from rag.openai_client import chat


def build_prompt(question: str, contexts: list[dict]) -> str:
    # Context with citations
    context_text = []
    for i, c in enumerate(contexts, start=1):
        citation = f"[{i}] {c['doc']} (page {c['page']})"
        context_text.append(f"{citation}\n{c['text']}\n")

    context_block = "\n---\n".join(context_text)

    return f"""
You are a precise RAG chatbot.

You MUST follow these rules:
1) Use ONLY the provided context. If the answer is not directly supported, say exactly:
   "I don't know based on the provided documents."
2) Do NOT use outside knowledge.
3) If you *must* infer something small from the context, add a separate section titled:
   "Adapting/Guessing:" and keep it to 1-2 sentences max.
4) Every factual sentence must end with citations like [1] or [2]. If you cannot cite it, you cannot say it.
5) First extract 2–4 short quotes (max 20 words each) from the context that support your answer.
   If you cannot find supporting quotes, you MUST answer: "I don't know based on the provided documents."

Output format:
- Quotes:
  - "..." [#]
- Answer:
- Grounding:
  - Used citations: [...]
  - If you answered "I don't know", say: Grounding: insufficient context
- Adapting/Guessing: (only if you inferred)

User question:
{question}

Context:
{context_block}
""".strip()


def answer_question(question: str, top_k: int = 5) -> str:
    contexts = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, contexts)
    return chat(prompt)


if __name__ == "__main__":
    q = "What is evidence based medicine?"
    print(answer_question(q, top_k=5))
