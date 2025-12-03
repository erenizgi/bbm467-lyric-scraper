const fs = require('fs');

function parseCsvArtistsTracksByName(filePath, artistCol = 'artists', trackCol = 'track_name') {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n').map(line => line.trim()).filter(Boolean);

  if (lines.length < 2) return []; // Boş dosya ise netekim

  // Header'ı ayıkla
  const header = lines[0].split(',').map(h => h.replace(/^"|"$/g, '').trim().toLowerCase());
  
  // Kolon indexlerini kap
  const artistIdx = header.findIndex(col => col === artistCol);
  const trackIdx  = header.findIndex(col => col === trackCol);

  if (artistIdx === -1 || trackIdx === -1) {
    throw new Error('Kolon isimleri dosyada yok!');
  }

  const result = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue; // Boş satırı atla
    const cols = lines[i].split(',');

    // Kolon sayısı eksikse geç
    if (cols.length <= Math.max(artistIdx, trackIdx)) continue;

    // Artist ve trackName’i isme göre bul
    let artistFull = cols[artistIdx].replace(/^"|"$/g, '').trim();
    let artistName = artistFull.split(';')[0].trim();
    let trackName = cols[trackIdx].replace(/^"|"$/g, '').trim();

    console.log(`Parsed line ${i}: Artist="${artistName}", Track="${trackName}"`);

    result.push([artistName, trackName]);
  }
  return result;
}

export default parseCsvArtistsTracksByName;