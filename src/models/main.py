import requests
import json
import time

def query_mistral(prompt, model="mistral:7b", max_tokens=100):
    url = "http://localhost:11434/api/generate"

    headers = {
        'Content-Type': 'application/json',
    }

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9
        }
    }

    try:
        print(f"Sending request to {url}")
        print(f"Request data: {json.dumps(data, indent=2)}")

        start_time = time.time()
        response = requests.post(url, headers=headers, data=json.dumps(data))
        end_time = time.time()

        print(f"\nResponse time: {end_time - start_time:.2f} seconds")
        print(f"Response status code: {response.status_code}")

        response.raise_for_status()

        result = response.json()
        return result['response']

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running."
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response from Ollama"

if __name__ == "__main__":
    response = query_mistral("What is 2+2?", max_tokens=50)
    print("\nFinal response:")
    print(response)