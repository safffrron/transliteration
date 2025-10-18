import json
import re
from openai import OpenAI
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
API_KEY = ""  # replace
MODEL = "nv-mistralai/mistral-nemo-12b-instruct"
DATA_FILE = "../hin/hin_test.json"  # path to your JSONL file

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

# ---------------------------
# HELPER: transliterate a word
# ---------------------------
def transliterate(word):
    prompt = f"""
You are a precise linguistic model for phonetic transliteration between English and Hindi (Devanagari).
Convert the English word below into its **exact Hindi script equivalent**, preserving pronunciation (not meaning).

Word: "{word}"

Output only the transliterated Hindi word — no explanation or extra text.
"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1,
        max_tokens=64,
        stream=False
    )
    
    # Extract output safely
    msg = completion.choices[0].message
    raw_output = getattr(msg, "content", "") or getattr(msg, "reasoning_content", "") or ""
    
    # Extract only Hindi characters (Devanagari)
    matches = re.findall(r"[\u0900-\u097F]+", raw_output)
    return "".join(matches) if matches else raw_output.strip()

# ---------------------------
# LOAD DATA
# ---------------------------
dataset = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

# ---------------------------
# EVALUATION
# ---------------------------
total_words = len(dataset)
word_correct = 0
total_chars = 0
char_correct = 0

for entry in tqdm(dataset, desc="Evaluating"):
    english_word = entry.get("english word")
    native_word = entry.get("native word")
    
    if not english_word or not native_word:
        continue
    
    pred_word = transliterate(english_word)
    
    # Word-level
    if pred_word == native_word:
        word_correct += 1
    
    # Character-level
    min_len = min(len(pred_word), len(native_word))
    for i in range(min_len):
        if pred_word[i] == native_word[i]:
            char_correct += 1
    total_chars += max(len(pred_word), len(native_word))

# ---------------------------
# RESULTS
# ---------------------------
word_acc = word_correct / total_words * 100
char_acc = char_correct / total_chars * 100

print(f"\nWord-level Accuracy: {word_acc:.2f}%")
print(f"Character-level Accuracy: {char_acc:.2f}%")
