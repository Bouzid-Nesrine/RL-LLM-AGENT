import MistralService from './MistralService';

class AppModel {
    constructor() {
        this.data = {
            messages: [],
            inputText: '',
            isTyping: false,
            themeColor: '#f8b6d2'
        };
        this.mistralService = new MistralService();
    }

    getData() {
        return this.data;
    }

    addMessage(message, isUser = false) {
        const newMessage = {
            id: Date.now(),
            text: message,
            isUser,
            timestamp: new Date().toLocaleTimeString()
        };
        this.data.messages = [...this.data.messages, newMessage];
        return this.data;
    }

    updateThemeColor(color) {
        this.data.themeColor = color;
        return this.data;
    }

    async getBotResponse(userMessage) {
        try {
            this.data.isTyping = true;
            const response = await this.mistralService.generateResponse(userMessage);
            this.data.isTyping = false;
            return response;
        } catch (error) {
            console.error('Error getting bot response:', error);
            this.data.isTyping = false;
            return "I apologize, but I'm having trouble generating a response right now. Please try again.";
        }
    }

    updateInputText(text) {
        this.data.inputText = text;
        return this.data;
    }
}

export default AppModel;