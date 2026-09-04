from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from retriever import retrieve
import os
from dotenv import load_dotenv

load_dotenv()


llm_model = ChatGroq(
    model_name = "openai/gpt-oss-120b",
    api_key = os.getenv("GROQ_LLM_API"),
    temperature = 0.2
)

def feedback(query):
    results = retrieve(query)
    context = "\n".join(doc.page_content for doc in results)
    combined_input = f"""
    Based on the following documents, answer the question: {query}
    Documents:
    {context}
    Provide a detailed and clear answer using the information from the documents.
    If the answer is not present in the documents, respond with "I couldn't find anything relevant in the loaded files."
    """

    messages = [
        SystemMessage(content = "You are a helpful assistant. Answer only from the provided documents."),
        HumanMessage(content = combined_input),
    ]

    return llm_model.invoke(messages).content


def start_chat():
    print("Welcome to RulesBot! Type 'exit' to quit")

    while True:
        question = input("Ask a question about the rules of the games: ").strip()
       
        if question.lower() in {"exit", "quit"}:
            print("Exiting chat. Goodbye!")
            break

        print("-----Generated Response------")
        print(feedback(question))

if __name__ == "__main__":
    start_chat()   