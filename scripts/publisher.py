import asyncio, os, sys, wave
import cv2, numpy as np
from livekit import rtc, api

os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"

URL, IDENTITY, VIDEO, WAV = sys.argv[1:5]   # ws://PI_IP:7880 pub1 face.mp4 face.wav
W, H, FPS = 640, 480, 30
SR, CH = 16000, 1
SAMPLES = SR // 100                          # 10 ms of audio per frame

async def pub_video(source):
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        sys.exit("cannot open video file")
    loop = asyncio.get_event_loop()
    nxt = loop.time()
    while True:
        ok, frame = cap.read()
        if not ok:                           # end of file: loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        rgba = cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2RGBA)
        source.capture_frame(
            rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, rgba.tobytes()))
        nxt += 1.0 / FPS
        await asyncio.sleep(max(0, nxt - loop.time()))

async def pub_audio(source):
    with wave.open(WAV, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    loop = asyncio.get_event_loop()
    nxt, i = loop.time(), 0
    while True:
        chunk = data[i:i + SAMPLES]
        if len(chunk) < SAMPLES:             # end of file: loop
            i = 0
            continue
        frame = rtc.AudioFrame.create(SR, CH, SAMPLES)
        np.copyto(np.frombuffer(frame.data, dtype=np.int16), chunk)
        await source.capture_frame(frame)
        i += SAMPLES
        nxt += 0.01
        await asyncio.sleep(max(0, nxt - loop.time()))

async def main():
    token = (api.AccessToken().with_identity(IDENTITY)
             .with_grants(api.VideoGrants(room_join=True, room="spike")).to_jwt())
    room = rtc.Room()
    await room.connect(URL, token)

    vsrc = rtc.VideoSource(W, H)
    vtrack = rtc.LocalVideoTrack.create_video_track("cam", vsrc)
    await room.local_participant.publish_track(
        vtrack, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))

    asrc = rtc.AudioSource(SR, CH)
    atrack = rtc.LocalAudioTrack.create_audio_track("mic", asrc)
    await room.local_participant.publish_track(
        atrack, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

    print(IDENTITY, "publishing")
    await asyncio.gather(pub_video(vsrc), pub_audio(asrc))

asyncio.run(main())
