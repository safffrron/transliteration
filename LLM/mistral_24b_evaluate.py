import json
import re
import os
from openai import OpenAI
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
API_KEY = ""
MODEL = "mistralai/mistral-small-3.1-24b-instruct-2503"
DATA_FILE = "../hin/hin_test.json"  # JSONL file
PROGRESS_FILE = "transliteration_progress.json"  # save progress

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
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1,
        max_tokens=64,
        stream=False
    )
    msg = completion.choices[0].message
    raw_output = getattr(msg, "content", "") or getattr(msg, "reasoning_content", "") or ""
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
# RESUME OPTION
# ---------------------------
start_index = 0
if os.path.exists(PROGRESS_FILE):
    choice = input(f"Found progress file '{PROGRESS_FILE}'. Resume? (y/n): ").strip().lower()
    if choice == "y":
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        start_index = progress.get("last_index", 0)
        word_correct = progress.get("word_correct", 0)
        char_correct = progress.get("char_correct", 0)
        total_chars = progress.get("total_chars", 0)
        print(f"Resuming from index {start_index}")
    else:
        word_correct = 0
        char_correct = 0
        total_chars = 0
else:
    word_correct = 0
    char_correct = 0
    total_chars = 0

total_words = len(dataset)

# ---------------------------
# EVALUATION LOOP
# ---------------------------
for idx in tqdm(range(start_index, total_words), desc="Evaluating"):
    entry = dataset[idx]
    english_word = entry.get("english word")
    native_word = entry.get("native word")

    if not english_word or not native_word:
        continue

    try:
        pred_word = transliterate(english_word)
    except Exception as e:
        print(f"Error on index {idx}, word '{english_word}': {e}")
        # Save progress before stopping
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "last_index": idx,
                "word_correct": word_correct,
                "char_correct": char_correct,
                "total_chars": total_chars
            }, f, ensure_ascii=False, indent=2)
        raise e

    # Word-level
    if pred_word == native_word:
        word_correct += 1

    # Character-level
    min_len = min(len(pred_word), len(native_word))
    for i in range(min_len):
        if pred_word[i] == native_word[i]:
            char_correct += 1
    total_chars += max(len(pred_word), len(native_word))

    # Save running progress every 10 words
    if idx % 10 == 0:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "last_index": idx,
                "word_correct": word_correct,
                "char_correct": char_correct,
                "total_chars": total_chars
            }, f, ensure_ascii=False, indent=2)

    # Print running accuracy
    current_word_acc = word_correct / (idx + 1) * 100
    current_char_acc = char_correct / total_chars * 100 if total_chars > 0 else 0
    tqdm.write(f"[{idx+1}/{total_words}] Running Word Acc: {current_word_acc:.2f}% | Char Acc: {current_char_acc:.2f}%")

# ---------------------------
# FINAL RESULTS
# ---------------------------
word_acc = word_correct / total_words * 100
char_acc = char_correct / total_chars * 100
print(f"\nFinal Word-level Accuracy: {word_acc:.2f}%")
print(f"Final Character-level Accuracy: {char_acc:.2f}%")

# Optionally delete progress file
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
