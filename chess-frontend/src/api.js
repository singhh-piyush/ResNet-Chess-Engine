const API_BASE = (import.meta.env?.VITE_API_BASE || '').replace(/\/$/, '');

export async function searchPosition(position, { signal, onProgress }) {
  let response;
  for (let attempt = 0; attempt < 5; attempt++) {
    response = await fetch(`${API_BASE}/predict/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(position), signal,
    });
    if (response.status !== 429 || attempt === 4) break;
    onProgress({ phase: 'waiting', message: 'Waiting for engine…' });
    await response.body?.cancel();
    await new Promise((resolve, reject) => {
      if (signal?.aborted) { reject(new DOMException('Search cancelled', 'AbortError')); return; }
      const abort = () => { clearTimeout(timer); reject(new DOMException('Search cancelled', 'AbortError')); };
      const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, 300 * 2 ** attempt);
      signal?.addEventListener('abort', abort, { once: true });
    });
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(response.status >= 500 ? 'The engine is unavailable. Retry in a moment.' : response.status === 429 ? 'The engine is busy. Retry in a moment.' : typeof body.detail === 'string' ? body.detail : 'The search could not start. Try again.');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, end); buffer = buffer.slice(end + 2);
        const event = frame.split('\n').find(line => line.startsWith('event: '))?.slice(7);
        const data = JSON.parse(frame.split('\n').filter(line => line.startsWith('data: ')).map(line => line.slice(6)).join('\n'));
        if (event === 'error') throw new Error(data.message);
        if (event === 'result') return data;
        if (event === 'started' || event === 'progress') onProgress({ ...data, phase: event });
      }
    }
    throw new Error('Search ended without a result');
  } finally { await reader.cancel().catch(() => {}); }
}
export { API_BASE };
