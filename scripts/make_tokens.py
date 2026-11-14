import os
from livekit import api

os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"

for identity in ["pub1", "pub2", "pub3", "pub4", "sub1"]:
    token = (
        api.AccessToken()
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room="spike"))
        .to_jwt()
    )
    print(identity, token)
