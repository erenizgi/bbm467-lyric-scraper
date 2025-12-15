const fs = require('fs');

// idCol parametresi eklendi (Varsayılan: 'original_id')
function parseCsvArtistsTracksByName(filePath, idCol = 'original_id', artistCol = 'artists', trackCol = 'track_name') {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n').map(line => line.trim()).filter(Boolean);

  if (lines.length < 2) return [];

  const header = lines[0].split(',').map(h => h.replace(/^"|"$/g, '').trim().toLowerCase());
  
  // ID, Artist ve Track indexlerini bul
  const idIdx = header.findIndex(col => col === idCol);
  const artistIdx = header.findIndex(col => col === artistCol);
  const trackIdx  = header.findIndex(col => col === trackCol);

  // ID kolonu zorunlu değil ama uyarı verelim, diğerleri zorunlu
  if (artistIdx === -1 || trackIdx === -1) {
    throw new Error('Artist veya Track kolonları dosyada yok!');
  }

  const result = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const cols = lines[i].split(',');

    if (cols.length <= Math.max(artistIdx, trackIdx)) continue;

    let artistFull = cols[artistIdx].replace(/^"|"$/g, '').trim();
    let artistName = artistFull.split(';')[0].trim();
    let trackName = cols[trackIdx].replace(/^"|"$/g, '').trim();
    
    // ID varsa al, yoksa 'UNKNOWN_ID' yaz
    let id = idIdx !== -1 && cols[idIdx] ? cols[idIdx].replace(/^"|"$/g, '').trim() : 'UNKNOWN_ID';

    console.log(`Parsed line ${i}: ID="${id}", Artist="${artistName}", Track="${trackName}"`);

    // Artık 3 veri döndürüyoruz: [id, artist, track]
    result.push([id, artistName, trackName]);
  }
  return result;
}

export default parseCsvArtistsTracksByName;