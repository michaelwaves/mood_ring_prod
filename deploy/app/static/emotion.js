export function getDominantEmotion(scores) {
    let maxEmotion = null;
    let maxScore = -Infinity;
    for (const [emotion, score] of Object.entries(scores)) {
        if (score > maxScore) {
            maxScore = score;
            maxEmotion = emotion;
        }
    }
    return maxEmotion;
}

export function getTopTwoEmotions(scores) {
    const sorted = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 2);
    return sorted.map(([emotion]) => emotion);
}

export function computeOverallEmotion(tokenScores) {
    if (tokenScores.length === 0) return null;

    const totals = {};
    for (const { scores } of tokenScores) {
        for (const [emotion, score] of Object.entries(scores)) {
            totals[emotion] = (totals[emotion] || 0) + score;
        }
    }

    return {
        dominant: getDominantEmotion(totals),
        topTwo: getTopTwoEmotions(totals),
        totals
    };
}

export function computeTopEmotionDistribution(tokenScores) {
    if (!tokenScores || tokenScores.length === 0) {
        return null;
    }

    const totals = {};
    let totalWeight = 0;

    for (const entry of tokenScores) {
        if (!entry || !entry.scores) continue;
        const dominant = getDominantEmotion(entry.scores);
        if (!dominant) continue;
        const score = entry.scores[dominant];
        const contribution = Number.isFinite(score) ? Math.max(score, 0) : 0;
        if (contribution <= 0) continue;
        totals[dominant] = (totals[dominant] || 0) + contribution;
        totalWeight += contribution;
    }

    if (totalWeight <= 0) {
        return null;
    }

    const breakdown = Object.entries(totals)
        .filter(([, value]) => value > 0)
        .map(([emotion, value]) => ({
            emotion,
            value,
            weight: value / totalWeight
        }))
        .sort((a, b) => b.value - a.value);

    return {
        totals,
        totalWeight,
        breakdown
    };
}
