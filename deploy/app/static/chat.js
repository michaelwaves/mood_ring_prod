import { escapeHtml } from './utils.js';

let chatHistory = [];
let currentAssistantMessage = null;
let currentTokenScores = [];

export function getChatHistory() {
    return chatHistory;
}

export function getCurrentAssistantMessage() {
    return currentAssistantMessage;
}

export function setCurrentAssistantMessage(message) {
    currentAssistantMessage = message;
}

export function getCurrentTokenScores() {
    return currentTokenScores;
}

export function setCurrentTokenScores(scores) {
    currentTokenScores = scores;
}

export function addToChatHistory(user, assistant = '') {
    chatHistory.push({ user, assistant });
}

export function updateLastAssistantMessage(content) {
    if (chatHistory.length > 0) {
        chatHistory[chatHistory.length - 1].assistant = content;
    }
}

export function resetTokenScores() {
    currentTokenScores = [];
}

export function addTokenScore(token, scores) {
    currentTokenScores.push({ token, scores });
}

export function addUserMessageToDOM(content, chatMessages) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(content)}</div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

export function addSystemMessageToDOM(content, chatMessages) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(content)}</div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

export function createAssistantMessageDOM(chatMessages) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="emotion-orb" title="Toggle emotion highlighting"></div>
        <div class="message-content">
            <div class="light-orb light-orb-1"></div>
            <div class="light-orb light-orb-2"></div>
            <div class="light-orb light-orb-3"></div>
            <div class="message-text"><span class="cursor"></span></div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}
