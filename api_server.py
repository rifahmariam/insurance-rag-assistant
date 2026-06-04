from fastapi import FastAPI, Request
import pandas as pd
import ollama
import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

customers = pd.read_csv("customer_database.csv")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    name = body.get("name") or request.query_params.get("name")
    question = body.get("question") or request.query_params.get("question")

    print("\n========== API CALLED ==========")
    print("Raw body:", body)
    print("Name:", name)
    print("Question:", question)
    print("================================\n")

    if not name or not question:
        return {"answer": "Please provide both customer name and question."}

    clean_name = re.sub(r"[^a-zA-Z]", "", name).lower()

    customers["clean_first_name"] = (
        customers["first_name"]
        .astype(str)
        .str.replace(r"[^a-zA-Z]", "", regex=True)
        .str.lower()
    )

    customer = customers[customers["clean_first_name"] == clean_name]

    if customer.empty:
        return {"answer": "Customer not found. Please check the name again."}

    provider = customer.iloc[0]["insurance_provider"].lower()
    first_name = customer.iloc[0]["first_name"]

    print("Customer found:", first_name)
    print("Provider:", provider)

    db_path = f"chroma_{provider}"

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    search_query = question

    if "waiting" in question.lower():
        search_query += (
            " waiting period pre-existing disease PED named ailments "
            "specific disease initial waiting period exclusions"
        )

    if "room" in question.lower() or "rent" in question.lower():
        search_query += (
            " room rent limit room category accommodation hospital room "
            "single private room ICU charges"
        )

    if "claim" in question.lower():
        search_query += (
            " claim settlement claim procedure number of claims cashless reimbursement"
        )

    if "exclusion" in question.lower():
        search_query += (
            " exclusions permanent exclusions non payable expenses"
        )

    docs = vectorstore.similarity_search(search_query, k=10)

    print("\nRetrieved Chunks:\n")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:1000])

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an insurance policy assistant.

Use only the retrieved policy context below.

Do not use general insurance knowledge.
Do not guess.
Do not apologize.
Do not mention technical issues.
Do not ask follow-up questions.

If the answer is present in the context, answer directly and clearly.
If the answer is not present in the context, say:
The requested information is not available in the policy documents.

Context:
{context}

Question:
{question}

Final answer:
"""

    response = ollama.chat(
        model="gemma:2b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"].strip()

    print("\nLLM ANSWER:\n")
    print(answer)
    print("\n=========================================\n")

    return {
        "name": first_name,
        "provider": provider,
        "answer": answer
    }