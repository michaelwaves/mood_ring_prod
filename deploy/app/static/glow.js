import { getEmotionColor, hexToRgbObj } from './colors.js';
import { getDominantEmotion, getTopTwoEmotions } from './emotion.js';

const DEFAULT_TRANSITION_DURATION = 4;
const HOLD_AFTER_TRANSITION = 1.2;
const CSS_TRANSITION_VAR = '--bg-transition-duration';
const LOOP_TOTAL_DURATION = 10;
const MIN_SEGMENT_DURATION = 1.2;

const glowStates = new WeakMap();

let currentOrbColor1 = { r: 0, g: 0, b: 0 };
let currentOrbColor2 = { r: 0, g: 0, b: 0 };
let orbElement = null;

function getTransitionDuration(element) {
    if (
        typeof window === 'undefined' ||
        typeof document === 'undefined' ||
        typeof getComputedStyle !== 'function'
    ) {
        return DEFAULT_TRANSITION_DURATION;
    }

    const target = element ?? document.documentElement;
    const raw = getComputedStyle(target).getPropertyValue(CSS_TRANSITION_VAR).trim();
    const parsed = parseFloat(raw);

    return Number.isFinite(parsed) ? parsed : DEFAULT_TRANSITION_DURATION;
}

function parseRgbTriple(value) {
    if (!value) return null;
    const matches = value.match(/[\d.]+/g);
    return matches && matches.length >= 3 ? matches : null;
}

function tripleToColor(triple) {
    return {
        r: parseFloat(triple[0]),
        g: parseFloat(triple[1]),
        b: parseFloat(triple[2]),
    };
}

function readColorFromStyles(element) {
    if (
        typeof window === 'undefined' ||
        typeof document === 'undefined' ||
        !element ||
        typeof getComputedStyle !== 'function'
    ) {
        return { r: 156, g: 163, b: 175 };
    }

    const styles = getComputedStyle(element);
    const rgbVar = styles.getPropertyValue('--aura-color-rgb').trim();
    const fromVar = parseRgbTriple(rgbVar);
    if (fromVar) {
        return tripleToColor(fromVar);
    }

    const fromBorder = parseRgbTriple(styles.borderColor);
    if (fromBorder) {
        return tripleToColor(fromBorder);
    }

    return { r: 156, g: 163, b: 175 };
}

function createGlowState(element) {
    return {
        element,
        currentColor: readColorFromStyles(element),
        queue: [],
        isProcessing: false,
        lastKey: null,
        currentTargetKey: null,
        holdTimeout: null,
        tween: null,
        looping: false,
        loopIndex: 0,
        loopSegments: [],
        loopDuration: LOOP_TOTAL_DURATION,
    };
}

function ensureGlowState(element) {
    if (!element) return null;
    let state = glowStates.get(element);
    if (!state) {
        state = createGlowState(element);
        glowStates.set(element, state);
    }
    return state;
}

function colorKey(rgb) {
    return `${Math.round(rgb.r)},${Math.round(rgb.g)},${Math.round(rgb.b)}`;
}

function cloneColor(rgb) {
    return { r: rgb.r, g: rgb.g, b: rgb.b };
}

function enqueueColor(state, rgb, options = {}) {
    if (!state) return;
    const {
        force = false,
        duration = null,
        holdDuration = HOLD_AFTER_TRANSITION,
    } = options;
    const key = colorKey(rgb);

    if (!force && (key === state.currentTargetKey || key === state.lastKey)) {
        return;
    }

    state.queue.push({
        rgb: cloneColor(rgb),
        key,
        duration,
        holdDuration,
    });
    if (!force) {
        state.lastKey = key;
    }

    processQueue(state);
}

function processQueue(state) {
    if (!state || state.isProcessing || state.queue.length === 0) {
        return;
    }

    state.isProcessing = true;
    const next = state.queue.shift();
    state.currentTargetKey = next.key;

    if (state.tween) {
        state.tween.kill();
        state.tween = null;
    }

    const transitionDuration = next.duration ?? getTransitionDuration(state.element);

    state.tween = gsap.to(state.currentColor, {
        r: next.rgb.r,
        g: next.rgb.g,
        b: next.rgb.b,
        duration: transitionDuration,
        overwrite: 'auto',
        onUpdate: () => {
            applyColor(state);
        },
        onComplete: () => {
            state.tween = null;
            if (state.holdTimeout) {
                clearTimeout(state.holdTimeout);
            }

            const holdDelay = Math.max(
                0,
                Number.isFinite(next.holdDuration)
                    ? next.holdDuration
                    : HOLD_AFTER_TRANSITION
            );

            if (holdDelay > 0) {
                state.holdTimeout = setTimeout(() => {
                    state.holdTimeout = null;
                    state.isProcessing = false;
                    state.currentTargetKey = null;

                    if (state.looping && state.queue.length === 0) {
                        scheduleNextLoopColor(state);
                    }

                    processQueue(state);
                }, holdDelay * 1000);
            } else {
                state.isProcessing = false;
                state.currentTargetKey = null;

                if (state.looping && state.queue.length === 0) {
                    scheduleNextLoopColor(state);
                }

                processQueue(state);
            }
        }
    });
}

function scheduleNextLoopColor(state) {
    if (!state.looping || state.loopSegments.length === 0) {
        return;
    }

    const segment = state.loopSegments[state.loopIndex];
    state.loopIndex = (state.loopIndex + 1) % state.loopSegments.length;

    const transitionDuration = getTransitionDuration(state.element);
    const targetTotal = Math.max(
        transitionDuration + MIN_SEGMENT_DURATION,
        segment.weight * state.loopDuration
    );
    const holdDuration = Math.max(
        MIN_SEGMENT_DURATION,
        targetTotal - transitionDuration
    );

    enqueueColor(state, segment.rgb, {
        force: true,
        duration: transitionDuration,
        holdDuration,
    });
}

function clearPending(state) {
    if (!state) return;
    if (state.tween) {
        state.tween.kill();
        state.tween = null;
    }
    if (state.holdTimeout) {
        clearTimeout(state.holdTimeout);
        state.holdTimeout = null;
    }
    state.isProcessing = false;
    state.currentTargetKey = null;
}

function applyColor(state) {
    if (!state || !state.element) return;

    const element = state.element;
    const r = Math.round(state.currentColor.r);
    const g = Math.round(state.currentColor.g);
    const b = Math.round(state.currentColor.b);

    const rgbString = `rgb(${r}, ${g}, ${b})`;
    const rgbaString = (alpha) => `rgba(${r}, ${g}, ${b}, ${alpha})`;

    element.style.setProperty('--aura-color-rgb', `${r}, ${g}, ${b}`);
    element.style.setProperty('--aura-color', rgbaString(0.5));
    element.style.borderColor = rgbaString(0.65);

    if (!element.classList.contains('highlighting')) {
        element.style.boxShadow = `
            inset 0 0 30px ${rgbaString(0.5)},
            inset 0 0 60px ${rgbaString(0.35)},
            inset 0 0 100px ${rgbaString(0.22)},
            0 4px 20px rgba(0, 0, 0, 0.35)
        `;
    }

    const orbs = element.querySelectorAll('.light-orb');
    orbs.forEach(orb => {
        orb.style.background = `linear-gradient(90deg, transparent 0%, ${rgbString} 50%, transparent 100%)`;
    });
}

export function updateGlow(content, scores) {
    if (!content) return;
    const dominant = getDominantEmotion(scores);
    if (!dominant) return;

    const color = getEmotionColor(dominant);
    const rgbColor = hexToRgbObj(color);

    const state = ensureGlowState(content);
    if (!state) return;

    if (state.looping) {
        return;
    }

    enqueueColor(state, rgbColor);
}

export function startGlowCycle(content, segments = []) {
    const state = ensureGlowState(content);
    if (!state) {
        return;
    }

    const normalized = (segments || [])
        .filter((segment) => segment && segment.rgb && segment.weight > 0)
        .map((segment) => ({
            rgb: cloneColor(segment.rgb),
            key: colorKey(segment.rgb),
            weight: segment.weight
        }));

    if (normalized.length === 0) {
        return;
    }

    state.looping = true;
    state.loopIndex = 0;
    state.loopSegments = normalized;
    state.queue = [];
    state.lastKey = null;
    clearPending(state);

    scheduleNextLoopColor(state);
}

export function updateOrbGlowWithTopTwo(orb, scores) {
    if (!orb || !scores) return;

    orbElement = orb;
    const topTwo = getTopTwoEmotions(scores);
    const transitionDuration = getTransitionDuration(orbElement);

    if (!topTwo || topTwo.length === 0) return;

    const color1 = getEmotionColor(topTwo[0]);
    const color2 = topTwo.length > 1 ? getEmotionColor(topTwo[1]) : color1;

    const rgb1 = hexToRgbObj(color1);
    const rgb2 = hexToRgbObj(color2);

    gsap.to(currentOrbColor1, {
        r: rgb1.r,
        g: rgb1.g,
        b: rgb1.b,
        duration: transitionDuration,
        onUpdate: () => {
            updateOrbStyles();
        }
    });

    gsap.to(currentOrbColor2, {
        r: rgb2.r,
        g: rgb2.g,
        b: rgb2.b,
        duration: transitionDuration,
        onUpdate: () => {
            updateOrbStyles();
        }
    });
}

export function updateOrbGradient(orb, scores) {
    updateOrbGlowWithTopTwo(orb, scores);
}

function updateOrbStyles() {
    if (!orbElement) return;

    const color1String = `rgb(${Math.round(currentOrbColor1.r)}, ${Math.round(currentOrbColor1.g)}, ${Math.round(currentOrbColor1.b)})`;
    const color2String = `rgb(${Math.round(currentOrbColor2.r)}, ${Math.round(currentOrbColor2.g)}, ${Math.round(currentOrbColor2.b)})`;

    orbElement.style.background = `linear-gradient(135deg, ${color1String} 0%, ${color2String} 100%)`;
    orbElement.style.boxShadow = `
        0 0 15px rgba(${Math.round(currentOrbColor1.r)}, ${Math.round(currentOrbColor1.g)}, ${Math.round(currentOrbColor1.b)}, 0.5),
        inset 0 0 10px rgba(255, 255, 255, 0.15)
    `;
}
