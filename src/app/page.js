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
      <main className="flex w-full max-w-4xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <h1 className='mb-4'>Lyric Hunter 🎤</h1>
        <form onSubmit={handleSearch} className='flex flex-col gap-4 w-fit'>
          <div className='flex gap-4 w-fit'>
            <input
            value={song}
            onChange={e => setSong(e.target.value)}
            placeholder="Song name"
            className="border-1 rounded-md border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <input
            value={artist}
            onChange={e => setArtist(e.target.value)}
            placeholder="Artist name"
            className="border-1 rounded-md border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"

          />
          <button
            type="submit"
            disabled={loading}
            className='border-1 min-w-32 rounded-md border-gray-300 px-4 py-2 flex items-center justify-center disabled:opacity-50'
            style={{textAlign: "center"}}
          >
            {loading ? 'Searching...' : 'Find!'}
          </button>
          </div>
          <div className="h-[100px] w-full flex items-center justify-center mt-4 text-center">
            {result && "Lyrics have been saved to lyrics_files folder!"}
          </div>
          
        </form>
        
        

      </main>
    </div>
  );
}
