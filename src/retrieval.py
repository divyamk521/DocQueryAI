from llm import get_llm
from prompt import get_prompt
from retriever import get_retriever
from chain import build_rag_chain

def run_retrieval():

    print("Initializing LLM and Retriever...")

    llm = get_llm()
    prompt = get_prompt()
    retriever = get_retriever()
    rag_chain = build_rag_chain(llm,
                                 prompt,
                                   retriever)

    print("\n--- Retrieval System Ready ---")
    
    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            break


       
        print(f"Searching for chunks related to: '{question}'...")

        docs = retriever.invoke(question)

        print(f"Found {len(docs)} matching chunks in PDF.")

        
        if docs :
            print("\nTop Retrieved Chunk:\n")
            print(docs[0].page_content[:300])


        response = rag_chain.invoke({
            "input": question
            })
        print(f"\nAI Answer: {response['answer']}\n")

if __name__ == "__main__":
    run_retrieval()