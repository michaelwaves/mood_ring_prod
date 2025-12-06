import { state, loadConfig } from './config.js';
import { getEmotionColor, hexToRgba, hexToRgbObj } from './colors.js';
import {
    getChatHistory,
    getCurrentAssistantMessage,
    setCurrentAssistantMessage,
    getCurrentTokenScores,
    addToChatHistory,
    updateLastAssistantMessage,
    resetTokenScores,
    addTokenScore,
    addUserMessageToDOM,
    addSystemMessageToDOM,
    createAssistantMessageDOM,
} from './chat.js';
import {
    getDominantEmotion,
    getTopTwoEmotions,
    computeOverallEmotion,
    computeTopEmotionDistribution,
} from './emotion.js';
import {
    buildLegend,
    setStatus,
    loadAboutContent,
    setupModalHandlers,
    setupInputHandlers,
    hideTooltip as hideUITooltip,
    setupTokenInteractions,
    setupHighlightToggle,
} from './ui.js';
import { connectWebSocket } from './websocket.js';
import { updateGlow, updateOrbGradient, startGlowCycle } from './glow.js';

function buildGlowCycleSegments(tokenScores) {
    const distribution = computeTopEmotionDistribution(tokenScores);
    if (!distribution || !distribution.breakdown.length) {
        return [];
    }

    return distribution.breakdown.slice(0, 3).map(({ emotion, weight }) => {
        const hex = getEmotionColor(emotion);
        const rgb = hexToRgbObj(hex);
        return { emotion, weight, rgb };
    });
}

let ws = null;
let isGenerating = false;

const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const legendContainer = document.getElementById('legend');
const aboutLink = document.getElementById('about-link');
const aboutModal = document.getElementById('about-modal');
const modalClose = document.getElementById('modal-close');
const emotionTooltip = document.getElementById('emotion-tooltip');

async function init() {
    await loadConfig();
    Object.assign(window, { config: state });

    loadAboutContent();
    buildLegend(legendContainer, state);

    ws = connectWebSocket(handleMessage);

    setupModalHandlers(aboutLink, modalClose, aboutModal);
    setupInputHandlers(userInput, chatForm, handleSubmit);

    chatMessages.addEventListener('scroll', () => hideUITooltip(emotionTooltip));
}

function handleMessage(data) {
    if (data.type === 'connected') {
        setStatus(statusIndicator, statusText, 'connected', 'Connected');
    } else if (data.type === 'disconnected') {
        setStatus(statusIndicator, statusText, 'disconnected', 'Disconnected');
    } else if (data.type === 'error') {
        if (data.message) {
            console.error('Server error:', data.message);
            addSystemMessageToDOM(`Error: ${data.message}`, chatMessages);
        }
        setGenerating(false);
    } else if (data.type === 'token') {
        handleToken(data);
    } else if (data.type === 'done') {
        handleDone();
    }
}

function handleToken(data) {
    if (data.token_count > 0) {
        addTokenScore(data.token, data.scores);
    }

    const message = getCurrentAssistantMessage();
    if (message) {
        const messageText = message.querySelector('.message-text');
        const content = message.querySelector('.message-content');

        const cursor = messageText.querySelector('.cursor');
        if (cursor) cursor.remove();

        const tokenSpan = document.createElement('span');
        tokenSpan.className = 'token-span';
        tokenSpan.textContent = data.token;

        const tokenIndex = getCurrentTokenScores().length - 1;
        tokenSpan.dataset.tokenIndex = tokenIndex;

        if (data.token_count > 0 && data.scores) {
            const dominant = getDominantEmotion(data.scores);
            if (dominant) {
                const color = getEmotionColor(dominant);
                tokenSpan.dataset.emotionColor = hexToRgba(color, 0.15);
            }
        }

        setupTokenInteractions(tokenSpan, tokenIndex, getCurrentTokenScores(), emotionTooltip);
        messageText.appendChild(tokenSpan);

        if (data.token_count > 0 && data.scores) {
            updateGlow(content, data.scores);

            const orb = message.querySelector('.emotion-orb');
            updateOrbGradient(orb, data.scores);
        }

        const newCursor = document.createElement('span');
        newCursor.className = 'cursor';
        messageText.appendChild(newCursor);

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function handleDone() {
    const message = getCurrentAssistantMessage();
    if (message) {
        const messageText = message.querySelector('.message-text');
        const content = message.querySelector('.message-content');
        const orb = message.querySelector('.emotion-orb');

        const cursor = messageText.querySelector('.cursor');
        if (cursor) cursor.remove();

        const overall = computeOverallEmotion(getCurrentTokenScores());
        if (overall && overall.dominant) {
            updateOrbGradient(orb, overall.totals);
        }

        const segments = buildGlowCycleSegments(getCurrentTokenScores());
        startGlowCycle(content, segments);

        updateLastAssistantMessage(messageText.textContent);
    }

    setGenerating(false);
}

function setGenerating(generating) {
    isGenerating = generating;
    sendBtn.disabled = generating;

    if (generating) {
        setStatus(statusIndicator, statusText, 'generating', 'Generating...');
    } else {
        setStatus(statusIndicator, statusText, 'connected', 'Ready');
        setCurrentAssistantMessage(null);
    }
}

function handleSubmit(e) {
    e.preventDefault();

    if (isGenerating || !ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }

    const message = userInput.value.trim();
    if (!message) return;

    addUserMessageToDOM(message, chatMessages);

    userInput.value = '';
    userInput.style.height = 'auto';

    resetTokenScores();

    const assistantMessage = createAssistantMessageDOM(chatMessages);
    setCurrentAssistantMessage(assistantMessage);

    const content = assistantMessage.querySelector('.message-content');
    const orb = assistantMessage.querySelector('.emotion-orb');
    setupHighlightToggle(content, orb);

    addToChatHistory(message);

    ws.send(JSON.stringify({
        message: message,
        history: getChatHistory().slice(0, -1),
    }));

    setGenerating(true);
}

document.addEventListener('DOMContentLoaded', init);
