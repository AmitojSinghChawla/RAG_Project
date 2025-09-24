from langchain.prompts import ChatPromptTemplate
from retrieval import vectordb, summarizer_llm, final_llm,retriever


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided documents to answer."),
    ("user", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    if not docs:
        return "No relevant documents found."
    return "\n\n".join([d.page_content for d in docs])

def answer_question(question):
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)
    final_input = qa_prompt.format_messages(question=question, context=context)
    response = final_llm.invoke(final_input)
    return response.content

if __name__ == "__main__":
    print("💬 Ask me anything about the documents (type 'exit' to quit)\n")
    while True:
        user_q = input("👉 Your question: ")
        if user_q.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        try:
            answer = answer_question(user_q)
            print("\n🤖 Answer:\n", answer, "\n" + "-"*100 + "\n")
        except Exception as e:
            print(f"⚠️ Error: {e}")
