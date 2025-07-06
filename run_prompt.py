import requests
import os
import json
from datetime import datetime
from config import OLLAMA_HOST, MODEL_NAME

def save_file(content, folder, filename):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved to {path}")

def generate_filename(base):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}.txt"

def run_query(prompt_text):
    print("Running prompt with Ollama...")
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "stream": False
        }
    )
    result = response.json()
    full_output = result.get("response", "")

    # Split outputs
    cot, json_block = "", ""
    if "```json" in full_output:
        cot, json_part = full_output.split("```json", 1)
        json_block = json_part.split("```")[0]
    else:
        cot = full_output

    # Save outputs
    save_file(prompt_text, "prompts", generate_filename("prompt"))
    save_file(cot.strip(), "logs", generate_filename("reasoning"))
    if json_block:
        save_file(json_block.strip(), "workflows", generate_filename("workflow")[:-4] + ".json")

if __name__ == "__main__":
    print("🔹 Geo-LLM Prompt Runner 🔹\n")
    prompt = input("Enter your geospatial query:\n> ")
    run_query(prompt)
