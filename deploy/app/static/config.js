export const EMOTION_COLORS = {
    love: '#FACC15',
    joy: '#22C55E',
    surprise: '#06B6D4',
    sadness: '#3B82F6',
    anger: '#D946EF',
    fear: '#EF4444',
    disgust: '#84CC16',
};

export const state = {
    emotions: ['joy', 'love', 'sadness', 'surprise', 'disgust'],
    colors: EMOTION_COLORS,
    model: 'Unknown',
    orthogonalize: 'none',
    projection: 'cos',
};

export async function loadConfig() {
    try {
        const response = await fetch('/config');
        const serverConfig = await response.json();
        Object.assign(state, serverConfig);

        if (serverConfig.colors) {
            Object.assign(state.colors, serverConfig.colors);
        }
    } catch (e) {
        console.warn('Could not load config, using defaults');
    }
}
