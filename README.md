
# 🩺 Medical Chatbot 

A lightweight, LLM-powered medical chatbot built with LangChain, Hugging Face embeddings, and a multi-agent AI system. Designed for preventive care, multilingual support, and real-time health document understanding.

⚠️ *Not for clinical diagnosis – informational use only.*

---

## 🚀 How to Run

Follow these steps to get started:

---

### 📁 Step 1: Clone the Repository

```bash
git clone https://github.com/ShubhamMallick/Medical-Chatbot.git
cd Medical-Chatbot
```

---

### 🐍 Step 2: Create and Activate a Conda Environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

---

### 📦 Step 3: Install the Requirements

```bash
pip install -r requirements.txt
```

In case of import issues:
```bash
pip install --upgrade sentence-transformers huggingface_hub
pip install langchain-community
```

---

### ▶️ Step 4: Run the Application

```bash
python app.py
```

---

## 🧠 Key Features

- ✅ **Document-Aware Chatbot**: Upload PDFs and ask questions based on the content.
- 🤖 **Agentic Architecture**: Profile, Risk, Insight, Guidance & Motivation agents.
- 🌐 **Multilingual Support**: Responds in local languages.
- 🧬 **Embeddings Powered by Hugging Face**: `all-MiniLM-L6-v2` for efficient vector search.
- 🔐 **Privacy-Aware**: Local processing, no data stored externally.

---

## 🙌 Credits

- Built with [LangChain](https://www.langchain.com/), [sentence-transformers](https://huggingface.co/sentence-transformers), and Microsoft AutoGen principles.
