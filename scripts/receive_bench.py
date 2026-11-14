"""receive_bench.py - LiveKit receive-path benchmark (no model yet).

Usage (on the Pi, inside ~/spike-env, with the publishers already running):
    python receive_bench.py <n_publishers_running> [run_seconds]
    python receive_bench.py 3 120

Each run writes its own CSV: receive_bench_<n>pub_<HHMMSS>.csv
"""
import asyncio
import csv
import os
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
import psutil
from livekit import api, rtc

os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"

URL = "ws://localhost:7880"
ROOM = "spike"

N_PUB = sys.argv[1] if len(sys.argv) > 1 else "1"          # label only, checked by n_streams
RUN_SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
INTERVAL = 10                                               # seconds per CSV row
KEEP_EVERY = 0.2                                            # target: 5 frames/s per participant
KEEP_CUTOFF = KEEP_EVERY * 0.9                              # 0.18 s: 25 fps arrival still gives 5 kept/s
OUT = f"receive_bench_{N_PUB}pub_{time.strftime('%H%M%S')}.csv"

# Unique identity every run, so two runs can never kick each other out of the room.
IDENTITY = f"sub-{N_PUB}-{uuid.uuid4().hex[:4]}"

stats = {}                  # participant identity -> {"recv", "kept", "ms": [this interval only]}
disconnected = asyncio.Event()


def other_instances():
    """Count other running copies of this script."""
    me = os.getpid()
    count = 0
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        if any("receive_bench.py" in part for part in (p.info["cmdline"] or [])):
            count += 1
    return count


def temp_c():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return ""


async def handle_video(stream, who):
    s = stats.setdefault(who, {"recv": 0, "kept": 0, "ms": []})
    last = 0.0
    async for ev in stream:
        s["recv"] += 1
        now = time.monotonic()
        if now - last < KEEP_CUTOFF:           # drop early, before any conversion
            continue
        last = now
        t0 = time.perf_counter()
        f = ev.frame.convert(rtc.VideoBufferType.RGB24)
        arr = np.frombuffer(f.data, np.uint8).reshape(f.height, f.width, 3)
        cv2.resize(arr, (224, 224))
        s["ms"].append((time.perf_counter() - t0) * 1e3)
        s["kept"] += 1


async def monitor(writer):
    psutil.cpu_percent(percpu=True)            # prime the counters
    prev = {}
    for _ in range(max(1, RUN_SECONDS // INTERVAL)):
        await asyncio.sleep(INTERVAL)
        if disconnected.is_set():
            print("Stopping early: this subscriber was disconnected.", flush=True)
            return
        cpu = psutil.cpu_percent(percpu=True)
        total = round(sum(cpu) / len(cpu), 1)
        temp = temp_c()
        now_str = time.strftime("%H:%M:%S")

        rates = {}
        for who, s in stats.items():
            pr, pk = prev.get(who, (0, 0))
            rates[who] = ((s["recv"] - pr) / INTERVAL, (s["kept"] - pk) / INTERVAL)
            prev[who] = (s["recv"], s["kept"])
        n_streams = sum(1 for r, _ in rates.values() if r > 0)

        if not stats:
            print(f"{now_str} no video tracks received yet - are the publishers running?", flush=True)
        for who, s in stats.items():
            recv, kept = rates[who]
            # Median of THIS interval's samples only; blank when nothing was converted.
            ms = round(float(np.median(s["ms"])), 2) if s["ms"] else ""
            s["ms"] = []
            row = [now_str, N_PUB, n_streams, who, round(recv, 1), round(kept, 1),
                   ms, total, cpu, temp]
            writer.writerow(row)
            print(row, flush=True)
        if stats and n_streams < int(N_PUB):
            print(f"WARNING: expected {N_PUB} streams but only {n_streams} delivered frames "
                  f"this interval.", flush=True)


async def main():
    if other_instances():
        sys.exit("Another receive_bench.py is already running. Stop it first "
                 "(pkill -f receive_bench) and start one run at a time.")

    token = (api.AccessToken().with_identity(IDENTITY)
             .with_grants(api.VideoGrants(room_join=True, room=ROOM)).to_jwt())
    room = rtc.Room()

    @room.on("track_subscribed")
    def on_sub(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print("subscribed to video from", participant.identity, flush=True)
            asyncio.ensure_future(
                handle_video(rtc.VideoStream(track), participant.identity))

    @room.on("disconnected")
    def on_disc(*args):
        print("DISCONNECTED from room:", args, flush=True)
        disconnected.set()

    await room.connect(URL, token)
    print(f"connected as {IDENTITY}; writing {OUT}", flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "n_pub_label", "n_streams", "who", "recv_fps", "kept_fps",
                    "convert_resize_ms", "cpu_total_pct", "cpu_per_core", "temp_c"])
        await monitor(w)

    print("participants that sent video:", sorted(stats), flush=True)
    await room.disconnect()


asyncio.run(main())
