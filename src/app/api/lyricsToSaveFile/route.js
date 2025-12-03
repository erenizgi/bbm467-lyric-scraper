import parseCsvArtistsTracksByName from "@/app/utils/csvSanitize";
import { fetchLyrics } from "@/app/utils/fetchLyrics";

export async function GET(request) {
    const { searchParams } = new URL(request.url);
    const fileName = searchParams.get('fileName');


    if (!fileName) {
        return Response.json({ error: 'File name is missing' }, { status: 400 });
    }
    const arr = parseCsvArtistsTracksByName(fileName, "artists", "track_name");
    console.log("Parsed length:", arr.length);


    const intervalCount = 500; // 500 ms between requests
    


    try {
        for (let i = 0; i < arr.length; i++) {
            const [artist, song] = arr[i];
            console.log(`\nProcessing ${i + 1}/${arr.length}: ${artist} - ${song}`);
            const res = await fetchLyrics(song, artist)
            if (res.error) {
                console.log(`Error fetching lyrics for ${artist} - ${song}: ${res.error}`);
            }else {
                console.log(`Fetched lyrics for ${artist} - ${song}`);
            }
            console.log("Status:", res.status);
        
            await new Promise(r => setTimeout(r, intervalCount));
            console.log(`Waiting for ${intervalCount} ms before next request...`);
        }
        
        return Response.json({
            status: "success",
        });

    } catch (err) {
        console.log('Error:', err);
        return Response.json({ error: err }, { status: 500 });
    }
}