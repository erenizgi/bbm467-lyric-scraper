'use client';

import { useState } from 'react';
export default function Home() {

  const [song, setSong] = useState('');
  const [artist, setArtist] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    const res = await fetch(`/api/lyrics?song=${encodeURIComponent(song)}&artist=${encodeURIComponent(artist)}`);
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex  w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <h1>Lyric Avcısı 🎤</h1>
        <form onSubmit={handleSearch}>
          <input
            value={song}
            onChange={e => setSong(e.target.value)}
            placeholder="Şarkı adı "
          />&nbsp;
          <input
            value={artist}
            onChange={e => setArtist(e.target.value)}
            placeholder="Sanatçı adı"
          />&nbsp;
          <button type="submit" disabled={loading}>{loading ? 'Aranıyor...' : 'Bul!'}</button>
        </form>
        {result && (
          <div style={{ marginTop: 24 }}>
            {result.url ? (
              <a href={result.url} target="_blank" rel="noopener noreferrer">Lyrics burada bak &#x1F449;</a>
            ) : (
              <div style={{ color: 'red' }}>{result.error}</div>
            )}
          </div>
        )}

      </main>
    </div>
  );
}
