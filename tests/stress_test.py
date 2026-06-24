import os
import requests
import time
import json
import sys
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Configuration pulled from .env
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/v1/chat/completions")
AUTH = (os.getenv("AUTH_USER"), os.getenv("AUTH_PASS"))
MODEL = os.getenv("MODEL")
TARGET_TOTAL_TOKENS = int(os.getenv("TARGET_TOTAL_TOKENS"))
CHUNK_LIMIT = int(os.getenv("MAX_TOKENS"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# The initial prompt designed to trigger a never-ending response
messages = [
    {"role": "system", "content": "You are a master world-builder and novelist. Your task is to write an incredibly detailed, never-ending sci-fi epic. Do not write summaries. Describe every room, every conversation, every technical mechanism, and every character's thoughts in agonizing detail."},
    {"role": "user", "content": "Begin the epic. Describe the awakening of a rogue AI on a derelict generation ship. Write as much as you possibly can."}
]

total_generated_tokens = 0
loop_count = 1
global_start_time = time.time()

print(f"=== INITIATING {TARGET_TOTAL_TOKENS} TOKEN STRESS TEST ===")
print(f"Model: {MODEL}")
print(f"Chunk Limit: {CHUNK_LIMIT} tokens\n")

try:
    while total_generated_tokens < TARGET_TOTAL_TOKENS:
        print(f"\n--- Starting Generation Loop {loop_count} ---")
        
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": CHUNK_LIMIT,
            "temperature": TEMPERATURE
        }

        response = requests.post(API_URL, json=payload, auth=AUTH, stream=True)
        response.raise_for_status()

        chunk_tokens = 0
        assistant_response = ""
        is_thinking = False
        loop_start_time = None

        # Stream the output
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        delta = data['choices'][0]['delta']
                        
                        # Handle reasoning tokens (thinking)
                        reasoning_text = delta.get('reasoning_content', '')
                        if reasoning_text:
                            if not is_thinking:
                                print("\033[90m", end='') # Dark grey for thinking
                                is_thinking = True
                            print(reasoning_text, end='', flush=True)
                            if loop_start_time is None: loop_start_time = time.time()
                            chunk_tokens += 1

                        # Handle actual story content
                        content_text = delta.get('content', '')
                        if content_text:
                            if is_thinking:
                                print("\033[0m", end='') # Reset color
                                is_thinking = False
                            print(content_text, end='', flush=True)
                            assistant_response += content_text
                            
                            if loop_start_time is None: loop_start_time = time.time()
                            chunk_tokens += 1
                            total_generated_tokens += 1
                            
                    except json.JSONDecodeError:
                        continue

        # Loop cleanup and metrics
        loop_time = time.time() - loop_start_time if loop_start_time else 0
        loop_tps = chunk_tokens / loop_time if loop_time > 0 else 0
        
        print(f"\n\033[93m[Loop {loop_count} Complete: {chunk_tokens} tokens | Speed: {loop_tps:.2f} TPS | Total Progress: {total_generated_tokens}/{TARGET_TOTAL_TOKENS}]\033[0m")
        
        # Append the assistant's response to the history so it remembers the story
        messages.append({"role": "assistant", "content": assistant_response})
        
        # Add a new user prompt telling it to keep going
        messages.append({"role": "user", "content": "Do not summarize. Continue the story exactly where you just left off, maintaining the excruciatingly high level of detail."})
        
        loop_count += 1

except KeyboardInterrupt:
    print("\n\nStress Test aborted by user.")
except Exception as e:
    print(f"\n\nNetwork or Server Error: {e}")
finally:
    total_time = time.time() - global_start_time
    avg_tps = total_generated_tokens / total_time if total_time > 0 else 0
    print("\n" + "="*50)
    print("=== FINAL STRESS TEST METRICS ===")
    print(f"Total Tokens Generated: {total_generated_tokens}")
    print(f"Total Elapsed Time:     {total_time / 60:.2f} minutes")
    print(f"Average Lifetime TPS:   {avg_tps:.2f} tokens/second")
    print("="*50)