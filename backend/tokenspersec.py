import requests
import time

model = "gpt-oss:20b"
prompt = "Explain how quantum computers work in detail. Be thorough and technical."

start_time = time.time()

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 4192  # Generate 512 tokens
        }
    }
)

end_time = time.time()
result = response.json()

eval_count = result["eval_count"]      # tokens generated
eval_duration = result["eval_duration"]  # nanoseconds

# Calculate tokens/second
tps = eval_count / (eval_duration / 1e9)

print(f"Tokens generated: {eval_count}")
print(f"Time: {end_time - start_time:.2f}s")
print(f"Tokens per second: {tps:.2f}")