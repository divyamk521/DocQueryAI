from retriever import get_retriever

retriever = get_retriever()

query = "What is diabetes?"

results = retriever.invoke(query)

print(f"Retrieved {len(results)} chunks\n")

for i, doc in enumerate(results):

    print(f"\nRESULT {i+1}")
    print("-" * 50)

    print(doc.page_content[:500])