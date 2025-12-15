import parseCsvArtistsTracksByName from "@/app/utils/csvSanitize";
import { fetchLyrics } from "@/app/utils/fetchLyrics";

export async function GET(request) {
    const { searchParams } = new URL(request.url);
    const fileName = searchParams.get('fileName');

    if (!fileName) {
        return Response.json({ error: 'File name is missing' }, { status: 400 });
    }

    // CSV okuyucu artık ID'yi de getiriyor
    const arr = parseCsvArtistsTracksByName(fileName, "original_id", "artists", "track_name");
    console.log("Parsed length:", arr.length);

    const intervalCount = 500; 

    try {
        for (let i = 0; i < arr.length; i++) {
            // Destructuring ile ID'yi de alıyoruz
            const [id, artist, song] = arr[i];

            console.log(`\nProcessing ${i + 1}/${arr.length}: [ID: ${id}] ${artist} - ${song}`);
            
            // ID'yi fetchLyrics'e gönderiyoruz
            const res = await fetchLyrics(song, artist, id);

            if (res.error) {
                console.log(`Error fetching lyrics for ${artist} - ${song}: ${res.error}`);
            } else {
                console.log(`Fetched lyrics for ${artist} - ${song}`);
            }
            console.log("Status:", res.status);
        
            await new Promise(r => setTimeout(r, intervalCount));
        }
        
        return Response.json({ status: "success" });

    } catch (err) {
        console.log('Error:', err);
        return Response.json({ error: err }, { status: 500 });
    }
}