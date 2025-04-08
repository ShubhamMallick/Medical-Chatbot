from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from dotenv import load_dotenv
from src.prompt import prompt_template
import os
import torch
import logging

# --- Flask app setup ---
app = Flask(__name__)

# --- Load environment variables ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not found in environment variables.")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# --- Setup logger ---
logging.basicConfig(level=logging.DEBUG)

# --- Load Hugging Face GPT-2 model and tokenizer ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# --- Create text generation pipeline with proper length controls ---
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=100,  # Focus on new tokens rather than total length
    do_sample=True,  # Enable sampling for more diverse outputs
    temperature=0.7,  # Control randomness
    top_p=0.9,  # Nucleus sampling
    pad_token_id=tokenizer.eos_token_id  # Ensure proper padding
)

# --- LangChain LLM wrapper ---
llm = HuggingFacePipeline(pipeline=generator)

# --- Load HuggingFace embeddings ---
embeddings = download_hugging_face_embeddings()

# --- Pinecone Vector Store ---
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# --- Retriever setup ---
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# --- Retrieval Augmented Generation (RAG) chain ---
prompt = ChatPromptTemplate.from_template(prompt_template)
qa_chain = create_stuff_documents_chain(llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# --- Flask routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "No message received", 400

    try:
        logging.debug(f"Received user message: {msg}")
        
        # Truncate input if too long (optional safety measure)
        max_input_length = 1024  # Adjust based on your model's limits
        if len(msg) > max_input_length:
            msg = msg[:max_input_length]
            logging.warning(f"Truncated long input to {max_input_length} characters")
        
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        
        # Clean up the response if needed
        answer = answer.strip()
        if answer.endswith(tokenizer.eos_token):
            answer = answer[:-len(tokenizer.eos_token)].strip()
            
        return str(answer)
    except Exception as e:
        logging.error(f"Error in processing message: {e}", exc_info=True)
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)