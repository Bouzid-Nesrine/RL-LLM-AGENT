class AppPresenter {
    constructor(model, view) {
        this.model = model;
        this.view = view;
        this.init();
    }

    init() {
        this.view.setPresenter(this);
        this.updateView();
    }

    updateView() {
        const data = this.model.getData();
        this.view.updateState(data);
    }

    async handleAction(action, payload) {
        switch (action) {
            case 'SEND_MESSAGE':
                if (payload.trim()) {
                    // Add user message
                    this.model.addMessage(payload, true);
                    this.updateView();
                    
                    // Clear input
                    this.model.updateInputText('');
                    this.updateView();

                    // Get bot response
                    const botResponse = await this.model.getBotResponse(payload);
                    this.model.addMessage(botResponse, false);
                    this.updateView();
                }
                break;

            case 'UPDATE_INPUT':
                this.model.updateInputText(payload);
                this.updateView();
                break;

            case 'UPDATE_THEME_COLOR':
                this.model.updateThemeColor(payload);
                this.updateView();
                break;

            default:
                console.warn('Unknown action:', action);
        }
    }
}

export default AppPresenter;
