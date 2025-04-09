from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAI
from dotenv import load_dotenv
from src.prompt import build_prompt
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import random
from pinecone import Pinecone, ServerlessSpec

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medicalbot"

# ✅ Initialize Pinecone client using v3.x
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Use Pinecone vectorstore properly
docsearch = LangchainPinecone(index, embeddings, "text")
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix: add pad token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)

    # Retrieve relevant context
    retrieved_docs = retriever.invoke(msg)
    raw_context = " ".join([doc.page_content.strip().replace("\n", " ") for doc in retrieved_docs])
    prompt = build_prompt(context=raw_context, question=msg)

    # Tokenize with proper padding
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    # Parse and clean generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answers = generated_text.split("Answer:")
    final_answer = answers[-1].strip() if len(answers) > 1 else generated_text.strip()

    print("🧠 Answer:", final_answer)
    return str(final_answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
