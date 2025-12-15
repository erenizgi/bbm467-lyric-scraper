import axios from 'axios';
import * as cheerio from 'cheerio';
import fs from 'fs';
import path from 'path';

// Dosya ismi temizleme fonksiyonunu buraya da ekleyelim (veya utils'den import edebilirsin)
function sanitize(str) {
  return str.replace(/[<>:"/\\|?*]+/g, '').trim();
}

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const song = searchParams.get('song');
  const artist = searchParams.get('artist');

  if (!song || !artist) {
    return Response.json({ error: 'Song or artist missing' }, { status: 400 });
  }
  
  const token = process.env.CLIENT_TOKEN;

  if (!token) {
    return Response.json({ error: 'NO API TOKEN!' }, { status: 500 });
  }

  try {
    
    const response = await axios.get('https://api.genius.com/search', {
      params: { q: `${song} ${artist}` },
      headers: {
        Authorization: `Bearer ${token}`
      }
    });

    const hit = response.data.response.hits[0];
    if (!hit) {
      return Response.json({ error: 'No songs found' }, { status: 404 });
    }

    console.log(`Found lyrics URL: ${hit.result.url}`);

    const pageResponse = await axios.get(hit.result.url);
    const $ = cheerio.load(pageResponse.data);  
    let lines = [];
    $('[data-lyrics-container="true"]').each((i, el) => {
      let currLine = "";
      $(el).contents().each((_, node) => {
        if (node.type === "text") {
          currLine += node.data;
        } else if (node.name === "br") {
          if (currLine.trim() !== "") {
            lines.push(currLine.trim());
            currLine = "";
          }
        }
      });
      if (currLine.trim() !== "") {
        lines.push(currLine.trim());
      }
    });

    if (!lines.length) {
      return Response.json({ error: 'Couldnt sanitize lyrics' }, { status: 404 });
    }
    const lyrics = lines.join('\n');

    // --- DEĞİŞİKLİK BURADA ---
    const safeArtist = sanitize(hit.result.primary_artist.name);
    const safeSong = sanitize(hit.result.title);
    
    // Manuel aramalar için ID'yi '0' veriyoruz ki Python kodu patlamasın.
    const manualId = '0'; 
    const fileName = `${manualId}_${safeSong}-${safeArtist}.txt`;
    
    const saveDir = path.join(process.cwd(), 'lyrics_files');
    const savePath = path.join(saveDir, fileName);
    
    if (!fs.existsSync(saveDir)) {
      fs.mkdirSync(saveDir, { recursive: true });
    }

    fs.writeFileSync(savePath, lyrics, 'utf8');
    console.log(`Lyrics saved: ${savePath}`);

    return Response.json({
      lyrics: lyrics,
      lines: lines,
    });

  } catch (err) {
    console.log('Error:', err);
    return Response.json({ error: err }, { status: 500 });
  }
}