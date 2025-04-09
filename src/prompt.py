
def build_prompt(context: str, question: str) -> str:
    clean_context = context[:500]  # Truncate for GPT-2 friendliness
    prompt = (
        f"Context: {clean_context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt
