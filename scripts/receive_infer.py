"""receive_infer.py - LiveKit receive + TFLite inference spike (CPU only).

Usage (Pi, inside ~/spike-env, publishers already running):
    python receive_infer.py <n_publishers> --video-model mobilenet_v2_1.0_224_quant.tflite
    python receive_infer.py 3 --video-model v.tflite --fps 2.5 --seconds 120

One CSV row per participant per 10 s: receive, sample, inference rate, drops,
stage times, score, whole-Pi CPU, temperature, throttle flag.
"""
import argparse
import asyncio
import csv
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import psutil
from livekit import api, rtc

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter

ap = argparse.ArgumentParser()
ap.add_argument("n_pub")
ap.add_argument("--video-model", required=True)
ap.add_argument("--audio-model", default=None)
ap.add_argument("--seconds", type=int, default=120)
ap.add_argument("--fps", type=float, default=5.0, help="target sampled frames/s per participant")
ap.add_argument("--workers", type=int, default=2, help="inference threads (1 thread each)")
args = ap.parse_args()

os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"
URL, ROOM = "ws://localhost:7880", "spike"
INTERVAL = 10
KEEP_CUTOFF = 0.9 / args.fps             # 0.18 s at 5 fps, 0.36 s at 2.5 fps
AUDIO_SR = 16000
AUDIO_N = 15600                          # YAMNet window, 0.975 s
MAX_INFLIGHT = args.workers * 2          # beyond this, frames/windows are dropped
OUT = f"infer_{args.n_pub}pub_{args.fps:g}fps_{time.strftime('%H%M%S')}.csv"
IDENTITY = f"sub-{args.n_pub}-{uuid.uuid4().hex[:4]}"

pool = ThreadPoolExecutor(max_workers=args.workers)
tl = threading.local()
inflight = {"n": 0}
stats = {}
disconnected = asyncio.Event()


def get_stats(who):
    return stats.setdefault(who, dict(recv=0, kept=0, done=0, dropped=0, prep=[], inf=[],
                                      scores=[], a_done=0, a_dropped=0, a_inf=[]))


def interp(kind, path):
    it = getattr(tl, kind, None)
    if it is None:
        it = Interpreter(model_path=path, num_threads=1)
        it.allocate_tensors()
        setattr(tl, kind, it)
    return it


def dequant(y, det):
    if y.dtype in (np.uint8, np.int8):
        scale, zp = det["quantization"]
        if scale:
            return (y.astype(np.float32) - zp) * scale
    return y.astype(np.float32)


def run_video(img):
    it = interp("v", args.video_model)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    x = img[None]
    if inp["dtype"] == np.float32:
        x = x.astype(np.float32) / 127.5 - 1.0
    t0 = time.perf_counter()
    it.set_tensor(inp["index"], x)
    it.invoke()
    y = dequant(it.get_tensor(out["index"])[0], out)
    return float(y.max()), (time.perf_counter() - t0) * 1e3


def run_audio(wave):
    it = interp("a", args.audio_model)
    inp = it.get_input_details()[0]
    outs = it.get_output_details()
    out = next((o for o in outs if o["shape"][-1] == 521), outs[0])
    t0 = time.perf_counter()
    it.set_tensor(inp["index"], wave.reshape(inp["shape"]).astype(np.float32))
    it.invoke()
    y = dequant(it.get_tensor(out["index"]), out)
    return float(y.max()), (time.perf_counter() - t0) * 1e3


def submit(fn, arg, on_done, on_drop):
    if inflight["n"] >= MAX_INFLIGHT:
        on_drop()
        return
    inflight["n"] += 1
    fut = asyncio.get_running_loop().run_in_executor(pool, fn, arg)

    def cb(f):
        inflight["n"] -= 1
        if f.exception():
            print("inference error:", f.exception(), flush=True)
        else:
            on_done(*f.result())
    fut.add_done_callback(cb)


async def handle_video(stream, who):
    s = get_stats(who)
    last = 0.0

    def done(score, ms):
        s["done"] += 1
        s["inf"].append(ms)
        s["scores"].append(score)

    def drop():
        s["dropped"] += 1

    async for ev in stream:
        s["recv"] += 1
        now = time.monotonic()
        if now - last < KEEP_CUTOFF:
            continue
        last = now
        t0 = time.perf_counter()
        f = ev.frame.convert(rtc.VideoBufferType.RGB24)
        arr = np.frombuffer(f.data, np.uint8).reshape(f.height, f.width, 3)
        small = cv2.resize(arr, (224, 224))
        s["prep"].append((time.perf_counter() - t0) * 1e3)
        s["kept"] += 1
        submit(run_video, small, done, drop)


async def handle_audio(stream, who):
    s = get_stats(who)
    buf = np.zeros(0, np.int16)

    def done(score, ms):
        s["a_done"] += 1
        s["a_inf"].append(ms)

    def drop():
        s["a_dropped"] += 1

    async for ev in stream:
        buf = np.concatenate([buf, np.frombuffer(ev.frame.data, np.int16)])
        while len(buf) >= AUDIO_N:
            wave = buf[:AUDIO_N].astype(np.float32) / 32768.0
            buf = buf[AUDIO_N:]
            submit(run_audio, wave, done, drop)


def sysinfo():
    temp, thr = "", ""
    try:
        temp = float(subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
                     .split("=")[1].split("'")[0])
        thr = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip().split("=")[1]
    except Exception:
        pass
    return temp, thr


def med(x, q=50):
    return round(float(np.percentile(x, q)), 2) if x else ""


async def monitor(writer):
    psutil.cpu_percent(percpu=True)
    prev = {}
    for _ in range(max(1, args.seconds // INTERVAL)):
        await asyncio.sleep(INTERVAL)
        if disconnected.is_set():
            print("Stopping early: disconnected.", flush=True)
            return
        cpu = psutil.cpu_percent(percpu=True)
        total = round(sum(cpu) / len(cpu), 1)
        temp, thr = sysinfo()
        now_str = time.strftime("%H:%M:%S")
        for who, s in stats.items():
            keys = ("recv", "kept", "done", "dropped", "a_done", "a_dropped")
            p = prev.get(who, dict.fromkeys(keys, 0))
            d = {k: (s[k] - p[k]) / INTERVAL for k in keys}
            prev[who] = {k: s[k] for k in keys}
            row = [now_str, args.n_pub, who, round(d["recv"], 1), round(d["kept"], 1),
                   round(d["done"], 1), round(d["dropped"], 1),
                   med(s["prep"]), med(s["inf"]), med(s["inf"], 95),
                   round(float(np.mean(s["scores"])), 3) if s["scores"] else "",
                   round(d["a_done"], 2), round(d["a_dropped"], 2), med(s["a_inf"]),
                   total, cpu, temp, thr]
            s["prep"], s["inf"], s["scores"], s["a_inf"] = [], [], [], []
            writer.writerow(row)
            print(row, flush=True)


async def main():
    token = (api.AccessToken().with_identity(IDENTITY)
             .with_grants(api.VideoGrants(room_join=True, room=ROOM)).to_jwt())
    room = rtc.Room()

    @room.on("track_subscribed")
    def on_sub(track, pub, participant):
        who = participant.identity
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print("video from", who, flush=True)
            asyncio.ensure_future(handle_video(rtc.VideoStream(track), who))
        elif track.kind == rtc.TrackKind.KIND_AUDIO and args.audio_model:
            print("audio from", who, flush=True)
            asyncio.ensure_future(handle_audio(
                rtc.AudioStream(track, sample_rate=AUDIO_SR, num_channels=1), who))

    @room.on("disconnected")
    def on_disc(*a):
        disconnected.set()

    await room.connect(URL, token)
    print(f"connected as {IDENTITY}; writing {OUT}", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "n_pub", "who", "recv_fps", "kept_fps", "vid_infer_fps",
                    "vid_dropped_fps", "prep_ms_med", "vid_infer_ms_med", "vid_infer_ms_p95",
                    "vid_score_mean", "aud_scores_per_s", "aud_dropped_per_s",
                    "aud_infer_ms_med", "cpu_total_pct", "cpu_per_core", "temp_c", "throttled"])
        await monitor(w)
    await room.disconnect()


asyncio.run(main())
