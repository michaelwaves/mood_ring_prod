export function connectWebSocket(onMessage) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        onMessage({ type: 'connected' });
    };

    ws.onclose = () => {
        onMessage({ type: 'disconnected' });
        setTimeout(() => connectWebSocket(onMessage), 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onMessage({ type: 'error' });
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        onMessage(data);
    };

    return ws;
}
