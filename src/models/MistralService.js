class MistralService {
    constructor() {
        this.API_URL = 'http://localhost:11434/api/generate';
    }

    async generateResponse(prompt, maxTokens = 100) {
        try {
            console.log('Sending request to Mistral via Ollama...', { prompt, maxTokens });
            const response = await fetch(this.API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: "mistral:7b",  // your local model name
                    prompt: prompt,
                    stream: false,
                    options: {
                        num_predict: maxTokens,
                        temperature: 0.7,
                        top_k: 40,
                        top_p: 0.9
                    }
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            console.log('Received response from Mistral:', data);
            return data.response;
        } catch (error) {
            console.error('Error in MistralService:', error);
            throw error;
        }
    }
}

export default MistralService;