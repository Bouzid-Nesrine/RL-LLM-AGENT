class OllamaService {
    constructor() {
        this.API_URL = 'http://localhost:11434/api/generate';
    }

    async generateResponse(prompt, maxTokens = 100) {
        try {
            console.log('Sending request to Ollama...', { prompt, maxTokens });
            const response = await fetch(this.API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: "llama3.2",
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
            console.log('Received response from Ollama:', data);
            return data.response;
        } catch (error) {
            console.error('Error in OllamaService:', error);
            throw error; // Re-throw to handle in AppModel
        }
    }
}

export default OllamaService;
