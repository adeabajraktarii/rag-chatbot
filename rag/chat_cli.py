from rag.rag_answer import answer_question

def main():
    print("📚 RAG Chatbot (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        print("\nBot:")
        print(answer_question(q, top_k=5))
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
