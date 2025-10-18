from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=""  
)

word = input()  # or take from input()

prompt = f"""
You are a precise linguistic model for phonetic transliteration between English and Hindi (Devanagari).
Your task: Convert the following English word into its exact Hindi script equivalent, preserving pronunciation (not meaning).

Word: "{word}"

Output only the transliterated Hindi word — no explanation, no punctuation, and no extra text.
"""

completion = client.chat.completions.create(
    model="nv-mistralai/mistral-nemo-12b-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=32,
    stream=False
)

output = completion.choices[0].message.content.strip()
print(output)
