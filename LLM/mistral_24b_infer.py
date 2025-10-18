import requests
import re

# ---------------------------
# CONFIG
# ---------------------------
invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
api_key = ""
word = input() # word to transliterate

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json"  # non-streaming for simplicity
}

# ---------------------------
# PROMPT
# ---------------------------
# prompt = f"""
# You are an expert linguistic model for phonetic transliteration between English and Hindi (Devanagari).

# Task:
# Convert the English word below into its exact Hindi script equivalent, preserving pronunciation (not meaning).

# Word: "{word}"

# Output only the transliterated Hindi word — no explanation, no punctuation, and no extra text.
# """

prompt = f"""
You are a precise linguistic model for phonetic transliteration between English and Hindi (Devanagari). 
Your task is to convert English words into Hindi script, preserving pronunciation exactly (do not translate the meaning). 
Output only the Hindi text — no explanations or extra words.

### Examples:

English: ram
Hindi: राम

English: shakti
Hindi: शक्ति

English: diwali
Hindi: दिवाली

English: krishna
Hindi: कृष्ण

English: birthday
Hindi: बर्थडे

English: janamdivas
Hindi: जन्मदिवस

English: rakha
Hindi: रक्खा

---

English: "{word}"
Hindi:
"""


payload = {
    "model": "mistralai/mistral-small-3.1-24b-instruct-2503",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 1,
    "stream": False
}

# ---------------------------
# CALL API
# ---------------------------
resp = requests.post(invoke_url, headers=headers, json=payload)
data = resp.json()

# Extract the model output
try:
    raw_output = data["choices"][0]["message"].get("content", "") or data["choices"][0]["message"].get("reasoning_content", "")
except Exception as e:
    print("⚠️ Error extracting output:", e)
    print("Full response:", data)
    exit()

# ---------------------------
# EXTRACT ONLY HINDI
# ---------------------------
matches = re.findall(r"[\u0900-\u097F]+", raw_output)
clean_word = "".join(matches) if matches else raw_output.strip()

print(clean_word)
