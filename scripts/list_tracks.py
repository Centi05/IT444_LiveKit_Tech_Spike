import asyncio, os
from livekit import rtc, api
os.environ["LIVEKIT_API_KEY"] = "devkey"; os.environ["LIVEKIT_API_SECRET"] = "secret"

async def main():
    t = (api.AccessToken().with_identity("peek")
         .with_grants(api.VideoGrants(room_join=True, room="spike")).to_jwt())
    room = rtc.Room()
    await room.connect("ws://localhost:7880", t)
    await asyncio.sleep(3)
    for ident, p in room.remote_participants.items():
        print(ident, "tracks:", len(p.track_publications))
    await room.disconnect()

asyncio.run(main())
