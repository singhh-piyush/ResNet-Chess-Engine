const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');

export async function searchPosition(position, { signal, onProgress }) {
  const response = await fetch(`${API_BASE}/predict/stream`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(position), signal,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Search failed (${response.status})`);
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
        if (event === 'started' || event === 'progress') onProgress(data);
      }
    }
    throw new Error('Search ended without a result');
  } finally { await reader.cancel().catch(() => {}); }
}
export { API_BASE };
