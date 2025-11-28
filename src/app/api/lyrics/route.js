import axios from 'axios';
import * as cheerio from 'cheerio';
import fs from 'fs';
import path from 'path';
import sanitize from '../sanitize';


export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const song = searchParams.get('song');
  const artist = searchParams.get('artist');

  if (!song || !artist) {
    return Response.json({ error: 'Şarkı veya sanatçı eksik bro' }, { status: 400 });
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
      return Response.json({ error: 'Şarkı bulunamadı' }, { status: 404 });
    }

    
    console.log(`Found lyrics URL:`);

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
            console.log(currLine.trim());
            currLine = "";
          }
        }
      });
      if (currLine.trim() !== "") {
        lines.push(currLine.trim());
      }
    });

    if (!lines.length) {
      return Response.json({ error: 'Lirik ayıklanamadı' }, { status: 404 });
    }
    const lyrics = lines.join('\n');



    const safeArtist = sanitize(artist);
    const safeSong = sanitize(song);
    const fileName = `${safeSong}-${safeArtist}.txt`;
    const saveDir = path.join(process.cwd(), 'lyrics_files');
    const savePath = path.join(saveDir, fileName);
    if (!fs.existsSync(saveDir)) {
      fs.mkdirSync(saveDir, { recursive: true });
    }

    fs.writeFileSync(savePath, lyrics, 'utf8');
    console.log(`Lirik kaydedildi: ${savePath} \n${lyrics}`);

    return Response.json({
      lyrics: lyrics,
      lines: lines,
    });

  } catch (err) {
    console.log('Hata:', err);
    return Response.json({ error: err }, { status: 500 });
  }
}