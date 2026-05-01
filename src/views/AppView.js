import React, { Component, createRef } from 'react';
import '../styles/App.css';

class AppView extends Component {
    constructor(props) {
        super(props);
        this.state = {
            messages: [],
            inputText: '',
            isTyping: false,
            themeColor: '#f8b6d2',
            showColorPicker: false,
            showEmojis: false
        };
        this.messagesEndRef = createRef();
        this.presenter = null;
        this.emojis = {
            'Favorites': ['😊', '😂', '❤️', '👍', '🎉', '✨', '🌟', '🔥', '💯', '🙌'],
            'Smileys': ['🥰', '😍', '😎', '🤗', '😋', '😄', '🥺', '😌', '😉', '🤩'],
            'Nature': ['🌸', '🌺', '🌹', '🌈', '⭐', '🌙', '☀️', '🌊', '🍀', '🦋']
        };
    }

    setPresenter(presenter) {
        this.presenter = presenter;
    }

    updateState(newState) {
        this.setState(newState, this.scrollToBottom);
        document.documentElement.style.setProperty('--theme-color', newState.themeColor);
    }

    scrollToBottom = () => {
        this.messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }

    handleInputChange = (e) => {
        this.presenter.handleAction('UPDATE_INPUT', e.target.value);
    }

    handleSubmit = (e) => {
        e.preventDefault();
        const message = this.state.inputText.trim();
        if (message) {
            this.presenter.handleAction('SEND_MESSAGE', message);
        }
    }

    handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleSubmit(e);
        }
    }

    handleColorChange = (e) => {
        if (this.presenter) {
            this.presenter.handleAction('UPDATE_THEME_COLOR', e.target.value);
            this.setState({ showColorPicker: false });
        }
    }

    toggleColorPicker = () => {
        this.setState(prevState => ({
            showColorPicker: !prevState.showColorPicker,
            showEmojis: false
        }));
    }

    toggleEmojis = () => {
        this.setState(prevState => ({
            showEmojis: !prevState.showEmojis,
            showColorPicker: false
        }));
    }

    addEmoji = (emoji) => {
        this.presenter.handleAction('UPDATE_INPUT', this.state.inputText + emoji);
    }

    render() {
        const { messages, inputText, isTyping, themeColor, showColorPicker, showEmojis } = this.state;

        return (
            <div className="app" style={{ '--theme-color': themeColor }}>
                <div className="chat-header">
                    <div className="header-left">
                        <div className="avatar">
                            <span>AI</span>
                        </div>
                        <div className="header-info">
                            <h1>AI Chatbot</h1>
                            {isTyping && <span className="status">typing...</span>}
                        </div>
                    </div>
                    <div className="header-actions">
                        <button 
                            onClick={this.toggleColorPicker}
                            className="action-button"
                            title="Change theme color"
                        >
                            🎨
                        </button>
                        {showColorPicker && (
                            <div className="popup-panel color-picker-popup">
                                <h3>Choose Theme Color</h3>
                                <input
                                    type="color"
                                    value={themeColor}
                                    onChange={this.handleColorChange}
                                    className="color-picker"
                                />
                                <div className="preset-colors">
                                    <button onClick={() => this.handleColorChange({ target: { value: '#f8b6d2' } })}>Pink</button>
                                    <button onClick={() => this.handleColorChange({ target: { value: '#b5a5f0' } })}>Purple</button>
                                    <button onClick={() => this.handleColorChange({ target: { value: '#a5e1f0' } })}>Blue</button>
                                    <button onClick={() => this.handleColorChange({ target: { value: '#f0d5a5' } })}>Orange</button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
                
                <div className="chat-messages">
                    <div className="welcome-message">
                        <h2>Welcome to AI Chat! 👋</h2>
                        <p>Ask me anything! I'm here to help.</p>
                    </div>
                    {messages.map(msg => (
                        <div key={msg.id} className={`message ${msg.isUser ? 'user' : 'bot'}`}>
                            <div className="message-bubble">
                                <div className="message-content">
                                    <p>{msg.text}</p>
                                    <span className="timestamp">{msg.timestamp}</span>
                                </div>
                            </div>
                        </div>
                    ))}
                    {isTyping && (
                        <div className="message bot">
                            <div className="message-bubble">
                                <div className="message-content">
                                    <div className="typing-indicator">
                                        <span></span>
                                        <span></span>
                                        <span></span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={this.messagesEndRef} />
                </div>

                <div className="chat-input-container">
                    <form onSubmit={this.handleSubmit} className="chat-input">
                        <button 
                            type="button" 
                            className="emoji-button"
                            onClick={this.toggleEmojis}
                            title="Add emoji"
                        >
                            <span role="img" aria-label="emoji">😊</span>
                        </button>
                        {showEmojis && (
                            <div className="popup-panel emoji-popup">
                                <div className="emoji-popup-header">
                                    <h3>Choose Emoji</h3>
                                    <button 
                                        className="close-emoji-popup"
                                        onClick={this.toggleEmojis}
                                        title="Close"
                                    >
                                        ×
                                    </button>
                                </div>
                                <div className="emoji-categories">
                                    {Object.entries(this.emojis).map(([category, categoryEmojis]) => (
                                        <div key={category} className="emoji-category">
                                            <h4>{category}</h4>
                                            <div className="emoji-grid">
                                                {categoryEmojis.map((emoji, index) => (
                                                    <button
                                                        key={`${category}-${index}`}
                                                        type="button"
                                                        onClick={() => this.addEmoji(emoji)}
                                                        className="emoji-item"
                                                        title={`Add ${emoji}`}
                                                    >
                                                        {emoji}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                        <textarea
                            value={inputText}
                            onChange={this.handleInputChange}
                            onKeyPress={this.handleKeyPress}
                            placeholder="Type your message..."
                            rows="1"
                        />
                        <button 
                            type="submit" 
                            className="send-button" 
                            disabled={!inputText.trim()}
                        >
                            <span className="button-content">
                                Send
                                <svg viewBox="0 0 24 24" className="send-icon">
                                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                                </svg>
                            </span>
                        </button>
                    </form>
                </div>
            </div>
        );
    }
}

export default AppView;
