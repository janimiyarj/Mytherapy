import os
import openai
from dotenv import load_dotenv
import logging
from rag.rag_engine import retrieve_relevant_chunks

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_advice(user_input: str) -> str:
    try:
        logger.info(f"[FALLBACK] Generating fallback advice for: {user_input}")

        prompt = (
    f"You are a compassionate and experienced mental health advisor.\n"
    f"The following is a concern shared by a patient:\n"
    f"\"{user_input}\"\n\n"
    "Please provide empathetic, supportive, and constructive advice that a counselor could use to help this patient.\n"
    "Ensure the response is gentle, encouraging, and prioritizes emotional well-being."
)


        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and compassionate mental health assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )

        logger.info("[LLM] Fallback response received successfully.")
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("GPT fallback error:", exc_info=True)
        return f"Error generating fallback advice: {str(e)}"



def generate_advice_with_rag(user_input: str) -> str:
    try:
        logger.info(f"[INPUT] User query: {user_input}")

        # Step 1: Vector DB lookup
        chunks = retrieve_relevant_chunks(user_input, top_k=3)
        logger.info(f"[RETRIEVED] Retrieved {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"[CHUNK {i}] {chunk[:250].strip()}...")

        if chunks:
            # Step 2: Format RAG prompt
            context = "\n".join(chunks)
            prompt = f"""You're a friendly and thoughtful mental health assistant.
The counselor is asking for support with a specific situation. Use the following real-life examples to gently guide your response.

Here’s what we know so far:
{context}

Here’s what the counselor shared:
{user_input}

Now, please offer kind, supportive, and practical advice that feels personal and emotionally sensitive.
Keep your tone warm and reassuring.
Advice:
"""


            logger.info(f"[PROMPT] Prompt to LLM:\n{prompt[:1000]}...")  # limit log length

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            logger.info("[LLM] RAG response received successfully.")
            return response.choices[0].message.content.strip()
        
        else:
            logger.warning("[FALLBACK] No relevant chunks found. Using fallback model.")
            return generate_advice(user_input)

    except Exception as e:
        logger.error(f" GPT RAG error: {e}")
        return f"Error generating advice: {str(e)}"



# (Optional) 🔧 Raw RAG + prompt helper for testing
def generate_rag_advice(user_input: str, docs: list) -> str:
    try:
        context = "\n".join(docs)
        prompt = f"""You’re a warm and friendly mental health assistant.
Based on the real-world context below, help a counselor who’s trying their best to support someone going through a tough time.

Context:
{context}

Here’s what the counselor said:
{user_input}

Now, kindly share heartfelt and encouraging guidance — something gentle, clear, and truly supportive.
Helpful Advice:
"""


        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("GPT direct RAG helper error:", e)
        return f"Error generating rag advice: {str(e)}"
