import { capitalize } from './utils.js';
import { getEmotionColor, hexToRgba } from './colors.js';
import { getDominantEmotion } from './emotion.js';

const LOADING_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

let statusAnimationInterval = null;

export function buildLegend(legendContainer, config) {
    if (!legendContainer) return;
    legendContainer.innerHTML = '';

    for (const emotion of config.emotions) {
        const color = config.colors[emotion] || '#666';
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <span class="legend-dot" style="background: ${color}"></span>
            <span>${capitalize(emotion)}</span>
        `;
        legendContainer.appendChild(item);
    }
}

export function setStatus(statusIndicator, statusText, status, text) {
    statusIndicator.className = 'status-indicator';

    if (statusAnimationInterval) {
        clearInterval(statusAnimationInterval);
        statusAnimationInterval = null;
    }

    if (status === 'connected') {
        statusIndicator.classList.add('connected');
        statusIndicator.textContent = '>';
    } else if (status === 'generating') {
        statusIndicator.classList.add('generating');
        let frame = 0;
        statusIndicator.textContent = LOADING_SPINNER[0];
        statusAnimationInterval = setInterval(() => {
            frame = (frame + 1) % LOADING_SPINNER.length;
            statusIndicator.textContent = LOADING_SPINNER[frame];
        }, 80);
    } else {
        statusIndicator.textContent = 'x';
    }
    statusText.textContent = text;
}

export function updateModalConfig(config) {
    const modalConfigModel = document.getElementById('modal-config-model');
    const modalConfigLayer = document.getElementById('modal-config-layer');
    const modalConfigProjection = document.getElementById('modal-config-projection');
    const modalConfigOrtho = document.getElementById('modal-config-ortho');

    const modelName = config.model ? config.model.split('/').pop() : '--';
    if (modalConfigModel) modalConfigModel.textContent = modelName;
    if (modalConfigLayer) modalConfigLayer.textContent = config.layer || 'Auto';
    if (modalConfigProjection) modalConfigProjection.textContent = config.projection === 'cos' ? 'Cosine' : 'Scalar';
    if (modalConfigOrtho) modalConfigOrtho.textContent = capitalize(config.orthogonalize || 'none');
}

export async function loadAboutContent() {
    try {
        const response = await fetch('/static/about.md');
        if (!response.ok) throw new Error('Failed to load about content');
        const text = await response.text();
        const contentDiv = document.getElementById('about-content');
        if (contentDiv && window.marked) {
            contentDiv.innerHTML = marked.parse(text);
            updateModalConfig(window.config);
        }
    } catch (e) {
        console.error('Error loading about content:', e);
    }
}

export function setupModalHandlers(aboutLink, modalClose, aboutModal) {
    if (aboutLink) {
        aboutLink.addEventListener('click', (e) => {
            e.preventDefault();
            aboutModal.classList.add('visible');
        });
    }

    if (modalClose) {
        modalClose.addEventListener('click', () => {
            aboutModal.classList.remove('visible');
        });
    }

    if (aboutModal) {
        aboutModal.addEventListener('click', (e) => {
            if (e.target === aboutModal) {
                aboutModal.classList.remove('visible');
            }
        });
    }
}

export function setupInputHandlers(userInput, chatForm, onSubmit) {
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSubmit(e);
        }
    });

    chatForm.addEventListener('submit', onSubmit);
}

export function hideTooltip(emotionTooltip) {
    emotionTooltip.classList.remove('visible');
}

export function showTooltip(emotionTooltip, event, tokenIndex, currentTokenScores) {
    if (tokenIndex < 0 || tokenIndex >= currentTokenScores.length) return;

    const tokenData = currentTokenScores[tokenIndex];
    const scores = tokenData.scores;

    const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);

    let html = `<div class="tooltip-title">Emotion Scores</div>`;
    for (const [emotion, score] of sorted) {
        const color = getEmotionColor(emotion);
        html += `
            <div class="tooltip-row">
                <span class="tooltip-emotion">
                    <span class="tooltip-dot" style="background: ${color}"></span>
                    ${capitalize(emotion)}
                </span>
                <span class="tooltip-score">${score.toFixed(3)}</span>
            </div>
        `;
    }

    emotionTooltip.innerHTML = html;
    emotionTooltip.classList.add('visible');
    moveTooltip(emotionTooltip, event);
}

export function moveTooltip(emotionTooltip, event) {
    const x = event.clientX + 15;
    const y = event.clientY + 15;

    const rect = emotionTooltip.getBoundingClientRect();
    const maxX = window.innerWidth - rect.width - 10;
    const maxY = window.innerHeight - rect.height - 10;

    emotionTooltip.style.left = Math.min(x, maxX) + 'px';
    emotionTooltip.style.top = Math.min(y, maxY) + 'px';
}

export function setupTokenInteractions(tokenSpan, tokenIndex, currentTokenScores, emotionTooltip) {
    tokenSpan.addEventListener('mouseenter', (e) => showTooltip(emotionTooltip, e, tokenIndex, currentTokenScores));
    tokenSpan.addEventListener('mouseleave', () => hideTooltip(emotionTooltip));
    tokenSpan.addEventListener('mousemove', (e) => moveTooltip(emotionTooltip, e));
}

export function setupHighlightToggle(content, orb) {
    orb.addEventListener('click', (e) => {
        e.stopPropagation();
        const isHighlighting = content.classList.toggle('highlighting');
        orb.classList.toggle('active', isHighlighting);

        const tokens = content.querySelectorAll('.token-span');
        tokens.forEach(token => {
            if (isHighlighting && token.dataset.emotionColor) {
                const baseColor = token.dataset.emotionColor.replace('0.15)', '1)');
                token.style.color = baseColor;
            } else {
                token.style.color = '';
            }
        });
    });
}
