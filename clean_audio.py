#!/usr/bin/env python3
"""
Audio cleanup pipeline: detect unwanted audio events and remove them.

Usage:
    python clean_audio.py input.mp3 output.mp3
    python clean_audio.py input.mp3 output.mp3 --events-csv events.csv
    python clean_audio.py input.mp3 output.mp3 --detect-only
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import tempfile
import unicodedata

import librosa
import numpy as np
import soundfile as sf


# Detection sensitivity presets: percentile thresholds for each feature.
# Lower percentiles flag more frames -> more (and longer) events detected.
SENSITIVITY_PRESETS = {
    "strict": {"rms": 86, "flat": 72, "zcr": 70, "roll": 62},   # current defaults
    "medium": {"rms": 82, "flat": 68, "zcr": 66, "roll": 58},
    "loose":  {"rms": 78, "flat": 64, "zcr": 62, "roll": 54},
}


def detect_events(audio_path, sr=16000, sensitivity="strict"):
    """Detect unwanted audio events using spectral analysis.

    sensitivity is one of SENSITIVITY_PRESETS ("strict", "medium", "loose").
    Looser settings flag more events.
    """
    thr = SENSITIVITY_PRESETS[sensitivity]

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    frame = 4096
    hop = 512

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, roll_percent=0.85)[0]

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    score = (
        (rms_db > np.percentile(rms_db, thr["rms"]))
        & (flat > np.percentile(flat, thr["flat"]))
        & (zcr > np.percentile(zcr, thr["zcr"]))
        & (roll > np.percentile(roll, thr["roll"]))
    )

    times = librosa.frames_to_time(np.arange(len(score)), sr=sr, hop_length=hop)

    events = []
    start = None
    for t, active in zip(times, score):
        if active and start is None:
            start = t
        elif not active and start is not None:
            dur = t - start
            if 0.4 <= dur <= 10:
                events.append((max(0, start - 1.0), t + 1.5))
            start = None

    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] < 2.0:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return merged


# AudioSet classes that represent unwanted noise events.
NOISE_CLASSES = ["Cough", "Throat clearing", "Sneeze"]

PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_LABELS_URL = ("http://storage.googleapis.com/us_audioset/youtube_corpus/"
                    "v1/csv/class_labels_indices.csv")


def _ensure_panns_data():
    """Download the PANNs checkpoint and label file if missing.

    panns_inference shells out to `wget`, which is often absent on macOS,
    so fetch them here with urllib instead.
    """
    import urllib.request

    data_dir = os.path.join(os.path.expanduser("~"), "panns_data")
    os.makedirs(data_dir, exist_ok=True)
    targets = [
        (os.path.join(data_dir, "class_labels_indices.csv"), PANNS_LABELS_URL, "labels"),
        (os.path.join(data_dir, "Cnn14_mAP=0.431.pth"), PANNS_CHECKPOINT_URL, "model checkpoint (~320MB)"),
    ]
    for path, url, what in targets:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f"Downloading PANNs {what} -> {path}")
        urllib.request.urlretrieve(url, path)


def detect_events_panns(audio_path, threshold=0.3, win=1.0, hop_sec=0.5,
                        pad=0.15, classes=None, extend_ratio=0.33):
    """Detect noise events with a pretrained AudioSet tagger (PANNs).

    Unlike the spectral heuristic, this recognises actual sound classes
    (cough, throat clearing, sneeze) rather than guessing from loudness
    and noisiness, so it is far more accurate on music. It also returns
    tight windows around each event, which the inpainting repair needs.

    threshold - minimum class probability to flag a window
    win       - analysis window length (s)
    hop_sec   - step between windows (s)
    pad       - padding added around each detected event (s)
    """
    _ensure_panns_data()
    from panns_inference import AudioTagging
    from panns_inference.config import labels

    classes = classes or NOISE_CLASSES
    idxs = [labels.index(c) for c in classes if c in labels]
    if not idxs:
        raise ValueError(f"No matching AudioSet classes for {classes}")

    sr = 32000  # PANNs is trained at 32kHz
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    at = AudioTagging(checkpoint_path=None, device="cpu")

    win_n = int(win * sr)
    hop_n = int(hop_sec * sr)
    starts = list(range(0, max(1, len(y) - win_n + 1), hop_n))
    # Cover the final partial window too, so an event in the last second
    # of a file isn't invisible.
    if starts and starts[-1] + win_n < len(y):
        starts.append(max(0, len(y) - win_n))
    y = np.pad(y, (0, max(0, (starts[-1] + win_n) - len(y))))

    # Batch windows through the model for speed.
    batch = 32
    scores = []
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        block = np.stack([y[s:s + win_n] for s in chunk])
        out = at.inference(block)[0]
        scores.extend(out[:, idxs].max(axis=1).tolist())
        if (i // batch) % 20 == 0:
            pct = 100 * min(i + batch, len(starts)) / len(starts)
            print(f"  scanning... {pct:.0f}%", flush=True)

    scores = np.array(scores)

    # Hysteresis: a window must clearly exceed the threshold to start an
    # event, but neighbouring windows only need to stay above a lower
    # bound to extend it. Avoids both false positives on music and
    # events truncated mid-cough.
    active = np.zeros(len(scores), dtype=bool)
    low = threshold * extend_ratio
    for c in np.where(scores >= threshold)[0]:
        active[c] = True
        k = c - 1
        while k >= 0 and scores[k] >= low:
            active[k] = True
            k -= 1
        k = c + 1
        while k < len(scores) and scores[k] >= low:
            active[k] = True
            k += 1

    # Group consecutive active windows into events.
    events = []
    run_start = None
    for k, on in enumerate(active):
        if on and run_start is None:
            run_start = k
        elif not on and run_start is not None:
            s = starts[run_start] / sr
            e = (starts[k - 1] + win_n) / sr
            events.append((max(0.0, s - pad), e + pad))
            run_start = None
    if run_start is not None:
        s = starts[run_start] / sr
        e = (starts[-1] + win_n) / sr
        events.append((max(0.0, s - pad), e + pad))

    # Merge events that nearly touch.
    merged = []
    for s, e in events:
        if merged and s - merged[-1][1] < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _write_audio(path, data, sr):
    """Write audio losslessly by default.

    The repair touches well under 1% of a recording, so re-encoding the
    whole thing to MP3 would impose generation loss on everything to fix
    almost nothing -- and splitting afterwards would encode a second
    time. FLAC (24-bit) keeps the untouched audio bit-exact. An explicit
    .mp3/.ogg path is still honoured for final delivery.
    """
    arr = data.T if getattr(data, "ndim", 1) > 1 else data
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".ogg", ".m4a"):
        print(f"  note: writing lossy {ext} -- use .flac to avoid generation loss")
        sf.write(path, arr, sr)
    else:
        sf.write(path, arr, sr, subtype="PCM_24")


def _lossless_path(path):
    """Redirect a lossy output path to FLAC, preserving the stem."""
    stem, ext = os.path.splitext(path)
    return stem + ".flac" if ext.lower() in (".mp3", ".ogg", ".m4a") else path


def save_events_csv(events, path):
    """Save detected events to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for s, e in events:
            writer.writerow([f"{s:.2f}", f"{e:.2f}"])


def load_events_csv(path):
    """Load events from an existing CSV file."""
    events = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                events.append((float(row[0]), float(row[1])))
    return events


def remove_events(audio_path, output_path, events, fade_ms=50):
    """Remove detected events from audio, applying short crossfades."""
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]

    fade_samples = int(sr * fade_ms / 1000)

    segments = []
    prev_end = 0
    for start_sec, end_sec in sorted(events):
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        if start_sample <= prev_end:
            prev_end = max(prev_end, end_sample)
            continue

        segment = y[:, prev_end:start_sample].copy()
        if segment.shape[1] > 2 * fade_samples:
            # Fade in at every splice joint (but not at the very start of
            # the file), otherwise a fade-to-zero meets an abrupt
            # full-level start and clicks.
            if prev_end > 0:
                segment[:, :fade_samples] *= np.linspace(0, 1, fade_samples)
            segment[:, -fade_samples:] *= np.linspace(1, 0, fade_samples)
        segments.append(segment)
        prev_end = end_sample

    if prev_end < y.shape[1]:
        segment = y[:, prev_end:].copy()
        if segment.shape[1] > fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            segment[:, :fade_samples] *= fade_in
        segments.append(segment)

    if not segments:
        print("Warning: all audio would be removed. Writing original file.")
        segments = [y]

    cleaned = np.concatenate(segments, axis=1)
    if cleaned.shape[0] == 1:
        cleaned = cleaned[0]

    _write_audio(output_path, cleaned, sr)

    original_dur = y.shape[1] / sr
    cleaned_dur = cleaned.shape[-1] / sr
    removed_dur = original_dur - cleaned_dur
    print(f"Original: {original_dur:.1f}s | Cleaned: {cleaned_dur:.1f}s | Removed: {removed_dur:.1f}s ({len(events)} events)")


def _localize_transient(y, sr, ev_s, ev_e, pad=0.06, min_len=0.15):
    """Narrow a flagged window down to the transient burst inside it.

    The tagger reports ~1-4s windows, but a cough is a ~0.4s burst. This
    finds the loud region within the window using a short-time energy
    envelope and returns just that span, so the repair touches as little
    audio as possible.
    """
    a, b = int(ev_s * sr), int(ev_e * sr)
    a, b = max(0, a), min(y.shape[1], b)
    if b - a < int(min_len * sr):
        return ev_s, ev_e

    seg = y[:, a:b].mean(axis=0)
    step = max(1, int(0.010 * sr))          # 10ms envelope
    frames = [np.abs(seg[i:i + step]).max() for i in range(0, len(seg) - step, step)]
    if len(frames) < 5:
        return ev_s, ev_e
    env = 20 * np.log10(np.array(frames) + 1e-12)

    # Baseline = the quiet part of the window; the burst is what rises
    # clearly above it.
    base = np.percentile(env, 25)
    peak = env.max()
    if peak - base < 6.0:                   # no clear transient; keep as-is
        return ev_s, ev_e
    thr = base + 0.35 * (peak - base)

    above = np.where(env >= thr)[0]
    if len(above) == 0:
        return ev_s, ev_e
    lo, hi = above[0], above[-1] + 1

    s = ev_s + max(0.0, lo * step / sr - pad)
    e = ev_s + min((b - a) / sr, hi * step / sr + pad)
    if e - s < min_len:                     # don't over-narrow
        mid = 0.5 * (s + e)
        s, e = mid - min_len / 2, mid + min_len / 2
    return max(ev_s, s), min(ev_e, e)


def inpaint_events(audio_path, output_path, events, fade_ms=30):
    """Repair events by spectral inpainting -- the highest-fidelity method.

    Rather than replacing the event window wholesale (which blanks out any
    singing or music underneath, causing an audible dropout), this works
    per time-frequency cell: for each STFT bin inside the event, the
    magnitude is capped at what the surrounding music predicts, and only
    the excess energy -- the cough itself -- is removed. Music and voice
    that continue through the event are preserved, so there is no dip in
    level.

    Phase is left untouched, which keeps the underlying material coherent.

    margin     - how far above the local maximum a bin must sit before it
                 counts as noise (1.0 = any excess; higher = more surgical)
    floor_gain - most any single bin may be attenuated (0.25 = -12dB max),
                 which prevents spectral holes and audible level dips
    """
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]

    n_fft = 2048
    hop = 512
    guard = 0.75           # seconds of clean context sampled either side
    fade = int(sr * fade_ms / 1000)

    events = sorted(events)
    smooth_k = np.array([0.25, 0.5, 0.25])

    repaired = y.copy()
    for idx, (ev_s, ev_e) in enumerate(events):
        # Narrow the flagged window down to the actual transient. PANNs
        # windows run 1-4s but a cough lasts ~0.4s, so repairing the whole
        # window would needlessly process seconds of good music.
        s_sec, e_sec = _localize_transient(y, sr, ev_s, ev_e)

        s_smp = max(0, int(s_sec * sr) - fade)
        e_smp = min(y.shape[1], int(e_sec * sr) + fade)
        if e_smp - s_smp < n_fft:
            continue

        # Guard regions: clean context either side, truncated at file
        # edges rather than dropped, and never overlapping a neighbouring
        # event (whose cough would corrupt the reference).
        g = int(guard * sr)
        pre_lo = max(0, s_smp - g)
        if idx > 0:
            pre_lo = max(pre_lo, min(s_smp, int(events[idx - 1][1] * sr)))
        post_hi = min(y.shape[1], e_smp + g)
        if idx + 1 < len(events):
            post_hi = min(post_hi, max(e_smp, int(events[idx + 1][0] * sr)))

        if pre_lo >= s_smp and e_smp >= post_hi:
            continue

        # One gain mask, computed from the mid signal and applied to both
        # channels, so the stereo image doesn't shift inside the event.
        mid = y[:, s_smp:e_smp].mean(axis=0)
        D_mid = librosa.stft(mid, n_fft=n_fft, hop_length=hop)
        mag_mid = np.abs(D_mid)

        def _ref_mag(lo, hi):
            if hi - lo < n_fft:
                return None
            r = y[:, lo:hi].mean(axis=0)
            return np.abs(librosa.stft(r, n_fft=n_fft, hop_length=hop))

        pre_ref = _ref_mag(pre_lo, s_smp)
        post_ref = _ref_mag(e_smp, post_hi)
        if pre_ref is None and post_ref is None:
            continue

        # Median (not max) is the robust estimate of what the context
        # typically holds in each band; max would be set by any transient
        # in the guard and would leave the cough untouched.
        pre_med = np.median(pre_ref, axis=1) if pre_ref is not None else None
        post_med = np.median(post_ref, axis=1) if post_ref is not None else None
        if pre_med is None:
            pre_med = post_med
        if post_med is None:
            post_med = pre_med

        # Interpolate the expected spectrum across the event so music that
        # evolves through it is tracked rather than flattened.
        n_frames = mag_mid.shape[1]
        ramp = np.linspace(0, 1, n_frames)[np.newaxis, :]
        target = pre_med[:, None] * (1 - ramp) + post_med[:, None] * ramp

        # Attenuate down to the expected spectrum with no floor: a cough
        # sits 25-35dB above quiet ambience, so a -12dB limit would only
        # make it quieter, not remove it.
        gain = np.ones_like(mag_mid)
        np.divide(target, mag_mid, out=gain, where=mag_mid > target)
        gain = np.clip(gain, 0.0, 1.0)

        # Smooth the gain, normalising by a smoothed all-ones matrix so
        # the kernel's edge taper doesn't attenuate the first/last frame.
        def _smooth(a, axis):
            if a.shape[axis] < 3:
                return a
            return np.apply_along_axis(lambda v: np.convolve(v, smooth_k, mode="same"), axis, a)
        norm = _smooth(_smooth(np.ones_like(gain), 1), 0)
        gain = _smooth(_smooth(gain, 1), 0) / np.maximum(norm, 1e-9)
        gain = np.clip(gain, 0.0, 1.0)

        for ch in range(y.shape[0]):
            seg = y[ch, s_smp:e_smp]
            D = librosa.stft(seg, n_fft=n_fft, hop_length=hop)
            out = librosa.istft(gain * np.abs(D) * np.exp(1j * np.angle(D)),
                                hop_length=hop, length=len(seg))

            # Crossfade the repaired segment back in at the edges.
            mix = np.ones(len(seg))
            if fade > 0 and len(seg) > 2 * fade:
                mix[:fade] = np.linspace(0, 1, fade)
                mix[-fade:] = np.linspace(1, 0, fade)
            repaired[ch, s_smp:e_smp] = out * mix + seg * (1 - mix)

        print(f"  Inpainted event {idx:03d}: {s_sec:.2f}s - {e_sec:.2f}s "
              f"(from {ev_s:.2f}-{ev_e:.2f})")

    out = repaired[0] if repaired.shape[0] == 1 else repaired
    _write_audio(output_path, out, sr)
    total = sum(e - s for s, e in events)
    print(f"Inpainted {len(events)} events ({total:.1f}s) -- level and duration preserved.")


def repair_events(audio_path, output_path, events, ctx=3.0, fade_ms=250):
    """Repair detected events by stem-swapping instead of cutting.

    For each event window, Demucs separates a padded snippet into
    vocals vs. accompaniment; the event range is then replaced with the
    accompaniment-only audio (music keeps playing, the cough/noise is
    gone). The rest of the recording is untouched, so there is no
    overall fidelity loss and no timing shift.

    ctx     - extra context (s) around each event given to Demucs
              (separation quality is poor at snippet edges)
    fade_ms - crossfade between original and repaired audio at the
              event boundaries
    """
    import shutil
    import subprocess

    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    n_samples = y.shape[1]

    events = sorted(events)

    with tempfile.TemporaryDirectory(prefix="stemswap_") as tmp:
        # 1. Write a padded snippet per event.
        snippets = []
        for i, (s_sec, e_sec) in enumerate(events):
            ws = max(0, int((s_sec - ctx) * sr))
            we = min(n_samples, int((e_sec + ctx) * sr))
            snip = y[:, ws:we]
            path = os.path.join(tmp, f"ev{i:03d}.wav")
            sf.write(path, snip.T if snip.shape[0] > 1 else snip[0], sr)
            snippets.append((path, ws, we))

        # 2. Separate all snippets in one Demucs run (model loads once).
        sep_dir = os.path.join(tmp, "sep")
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "-o", sep_dir,
        ] + [p for p, _, _ in snippets]
        print(f"Separating {len(snippets)} event snippets with Demucs...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Demucs separation failed. Is demucs installed in this venv? "
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # 3. Patch each event range with the accompaniment-only audio.
        fade = int(sr * fade_ms / 1000)
        repaired = y.copy()
        for i, ((path, ws, we), (s_sec, e_sec)) in enumerate(zip(snippets, events)):
            name = os.path.splitext(os.path.basename(path))[0]
            music_path = os.path.join(sep_dir, "htdemucs", name, "no_vocals.wav")
            music, _ = librosa.load(music_path, sr=sr, mono=False)
            if music.ndim == 1:
                music = music[np.newaxis, :]
            if music.shape[0] != y.shape[0]:
                music = np.tile(music.mean(axis=0, keepdims=True), (y.shape[0], 1))

            # Event range within the snippet (offset by window start).
            es = int(s_sec * sr) - ws
            ee = int(e_sec * sr) - ws
            es = max(0, es)
            ee = min(music.shape[1], we - ws, ee)
            if ee <= es:
                continue

            # Crossfade original -> music at entry, music -> original at exit,
            # staying inside the context margins.
            fi = min(fade, es)            # entry fade length
            fo = min(fade, music.shape[1] - ee)  # exit fade length

            patch = music[:, es - fi:ee + fo].copy()
            dst_s = ws + es - fi
            dst_e = ws + ee + fo
            orig = repaired[:, dst_s:dst_e]

            mix = np.ones(patch.shape[1])
            if fi > 0:
                mix[:fi] = np.linspace(0, 1, fi)
            if fo > 0:
                mix[-fo:] = np.linspace(1, 0, fo)
            repaired[:, dst_s:dst_e] = patch * mix + orig * (1 - mix)
            print(f"  Repaired event {i:03d}: {s_sec:.2f}s - {e_sec:.2f}s")

    out = repaired[0] if repaired.shape[0] == 1 else repaired
    _write_audio(output_path, out, sr)
    total = sum(e - s for s, e in events)
    print(f"Repaired {len(events)} events ({total:.1f}s) in place -- duration unchanged.")


def _musical_change_points(y, sr, res=1.0, half=25.0, prom=0.25,
                           min_gap=90.0, ratio=1.8, ctx=45.0):
    """Find where one song gives way to another inside a musical passage.

    Consecutive songs are often played back to back with no talking or
    pause between them, so the Music-vs-Speech split leaves them merged.
    A new song shows up as a change in timbre (different voice, different
    language) and harmony (different melody, different key), which is
    what MFCC + chroma capture.

    Foote novelty over a self-similarity matrix proposes candidates; each
    is then kept only if the audio either side really differs by more
    than `ratio` times the track's own typical variation, which rejects
    peaks caused by ordinary variation within one song.

    Returns offsets in seconds from the start of `y`.
    """
    from scipy.signal import find_peaks

    hop = 1024
    fps = sr / hop
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)

    def _z(x):
        return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)

    feat = np.vstack([_z(mfcc), _z(chroma) * 1.5])

    # Aggregate to ~1s frames: song boundaries are a coarse structure and
    # note-level detail only adds noise.
    step = max(1, int(round(res * fps)))
    n = (feat.shape[1] // step) * step
    if n < step * 4:
        return []
    coarse = feat[:, :n].reshape(feat.shape[0], -1, step).mean(axis=2)
    unit = coarse / (np.linalg.norm(coarse, axis=0, keepdims=True) + 1e-9)

    sim = unit.T @ unit
    L = int(half / res)
    if sim.shape[0] < 2 * L + 2:
        return []

    taper = np.outer(np.hanning(2 * L), np.hanning(2 * L))
    kern = np.kron(np.array([[1.0, -1.0], [-1.0, 1.0]]), np.ones((L, L))) * taper
    kern /= np.abs(kern).sum()

    nov = np.zeros(sim.shape[0])
    for i in range(L, sim.shape[0] - L):
        nov[i] = (sim[i - L:i + L, i - L:i + L] * kern).sum()
    nov = np.maximum(nov, 0)
    if nov.max() <= 0:
        return []
    nov /= nov.max()

    peaks, _ = find_peaks(nov, prominence=prom, distance=int(min_gap / res))
    if not len(peaks):
        return []

    # Validate: compare the flanking windows against the track's own
    # typical window-to-window variation.
    def _vec(a, b):
        i, j = int(a * fps), int(b * fps)
        i, j = max(0, i), min(feat.shape[1], j)
        if j - i < 10:
            return None
        v = feat[:, i:j].mean(axis=1)
        return v / (np.linalg.norm(v) + 1e-9)

    total = feat.shape[1] / fps
    base = []
    for t in np.arange(ctx, max(ctx, total - 2 * ctx), ctx):
        u, v = _vec(t - ctx, t), _vec(t, t + ctx)
        if u is not None and v is not None:
            base.append(1 - float(u @ v))
    baseline = float(np.median(base)) if base else 0.0

    kept = []
    for p in peaks:
        t = p * res
        u, v = _vec(t - ctx, t), _vec(t, t + ctx)
        if u is None or v is None:
            continue
        d = 1 - float(u @ v)
        if baseline <= 0 or d / baseline >= ratio:
            kept.append(t)
    return kept


def split_songs(audio_path, out_dir, min_song=45.0, pad=1.0,
                win=1.0, hop_sec=1.0, music_only=False, subdivide=True):
    """Split a concert recording into individual songs.

    Songs at a live show are separated by applause and talking, not by
    silence, so the silence-based splitter misses those boundaries. This
    scores each second of the recording for Music vs. Applause/Speech
    using the same AudioSet tagger as detection, then cuts where music
    gives way to a sustained non-music stretch.

    min_song - discard segments shorter than this (s)
    pad      - keep this much audio either side of each boundary (s)
    """
    _ensure_panns_data()
    from panns_inference import AudioTagging
    from panns_inference.config import labels

    music_idx = [labels.index(c) for c in ["Music"] if c in labels]
    break_idx = [labels.index(c) for c in
                 ["Applause", "Speech", "Cheering", "Clapping",
                  "Hubbub, speech noise, speech babble"] if c in labels]

    sr = 32000
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    at = AudioTagging(checkpoint_path=None, device="cpu")

    win_n, hop_n = int(win * sr), int(hop_sec * sr)
    starts = list(range(0, max(1, len(y) - win_n + 1), hop_n))

    music, brk = [], []
    batch = 32
    for i in range(0, len(starts), batch):
        block = np.stack([y[s:s + win_n] for s in starts[i:i + batch]])
        out = at.inference(block)[0]
        music.extend(out[:, music_idx].max(axis=1).tolist())
        brk.extend(out[:, break_idx].max(axis=1).tolist())
        if (i // batch) % 20 == 0:
            print(f"  scanning... {100 * min(i + batch, len(starts)) / len(starts):.0f}%", flush=True)

    music, brk = np.array(music), np.array(brk)

    # Tags flicker second to second, so smooth over a long window: what
    # matters is whether a whole passage is musical, not any one second.
    smooth_n = max(3, int(round(20.0 / hop_sec)))
    kern = np.ones(smooth_n) / smooth_n
    music_s = np.convolve(music, kern, mode="same")
    brk_s = np.convolve(brk, kern, mode="same")

    is_music = music_s > brk_s

    # Boundaries are sustained transitions between musical and
    # non-musical passages -- short excursions are ignored.
    min_run = max(1, int(round(min_song * 0.5 / hop_sec)))
    runs = []
    run_start = 0
    for k in range(1, len(is_music)):
        if is_music[k] != is_music[k - 1]:
            runs.append((run_start, k, is_music[k - 1]))
            run_start = k
    runs.append((run_start, len(is_music), is_music[-1]))

    # Drop runs too short to be a real passage, then merge neighbours
    # that share a label -- otherwise a brief applause swell inside a
    # song would leave two adjacent "music" runs and cut mid-song.
    stable = [list(r) for r in runs if r[1] - r[0] >= min_run]
    merged_runs = []
    for r in stable:
        if merged_runs and merged_runs[-1][2] == r[2]:
            merged_runs[-1][1] = r[1]
        else:
            merged_runs.append(r)

    # Each surviving run is a passage; its label says whether it is
    # music. Segments must carry that label -- the passages *between*
    # songs are spoken, and writing them as "song_NN" would be wrong.
    if not merged_runs:
        print("No passages found. Try lowering --min-song.")
        return

    segments = [(starts[r[0]], starts[r[1]] if r[1] < len(starts) else len(y), r[2])
                for r in merged_runs]

    orig, osr = librosa.load(audio_path, sr=None, mono=False)
    if orig.ndim == 1:
        orig = orig[np.newaxis, :]
    scale = osr / sr  # map PANNs-rate indices back to the original rate

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    pad_n = int(pad * osr)

    # Consecutive songs often run together with no talking between them,
    # so subdivide each musical passage where the melody or language
    # changes.
    pieces = []
    for a, b, musical in segments:
        if (b - a) / sr < min_song:
            continue
        if not musical and music_only:
            continue
        if not musical or not subdivide:
            pieces.append((a, b, musical))
            continue

        lo, hi = int(a * scale), int(b * scale)
        mono = orig[:, lo:hi].mean(axis=0)
        an_sr = 22050
        mono = librosa.resample(mono, orig_sr=osr, target_sr=an_sr)
        cps = _musical_change_points(mono, an_sr)
        cps = [c for c in cps if c > min_song and (b - a) / sr - c > min_song]
        if cps:
            print(f"  subdividing {(b - a) / sr:.0f}s passage at "
                  + ", ".join(f"{c / 60:.1f}min" for c in cps))
        prev = a
        for c in cps:
            cut = a + c * sr          # back to PANNs-rate index space
            pieces.append((prev, cut, True))
            prev = cut
        pieces.append((prev, b, True))

    n_song = n_talk = 0
    for a, b, musical in pieces:
        s = max(0, int(a * scale) - pad_n)
        e = min(orig.shape[1], int(b * scale) + pad_n)
        seg = orig[:, s:e]
        if seg.shape[0] == 1:
            seg = seg[0]
        if musical:
            n_song += 1
            path = os.path.join(out_dir, f"{base}_song_{n_song:02d}.flac")
            kind = f"Song {n_song:02d}"
        else:
            n_talk += 1
            path = os.path.join(out_dir, f"{base}_talk_{n_talk:02d}.flac")
            kind = f"Talk {n_talk:02d}"
        _write_audio(path, seg, osr)
        print(f"  {kind}: {s / osr:8.1f}s - {e / osr:8.1f}s  ({(e - s) / osr:6.1f}s)  -> {path}")

    if n_song == 0 and n_talk == 0:
        print("No passages long enough. Try lowering --min-song.")
    else:
        print(f"Wrote {n_song} songs and {n_talk} spoken passages to {out_dir}/")


# Languages worth trusting a transcript from. A detection outside this
# set usually means the singing is non-lexical (vocables), not that the
# recording is in an exotic language.
TITLE_LANGS = {"en", "es", "pt", "hu"}

_JUNK_WORD = re.compile(r"^(bleh|arrgh|uh+|ah+|oh+|hmm+|mm+|la|na|ha)$", re.I)


def _clean_title(text, max_words=6):
    text = re.sub(r"[^\w\s'\-]", " ", text, flags=re.UNICODE)
    words = [w for w in text.split() if w]
    if not words:
        return None
    title = " ".join(words[:max_words]).strip()
    return title[:1].upper() + title[1:] if title else None


def _titleable(text, lang):
    """Is this transcript real lyrics, or noise the model invented?"""
    words = [w.lower() for w in re.findall(r"[^\W\d_]+", text, re.UNICODE)]
    if len(words) < 8:
        return False
    if len(set(words)) / len(words) < 0.25:        # same token over and over
        return False
    if sum(bool(_JUNK_WORD.match(w)) for w in words) > 0.4 * len(words):
        return False
    return lang in TITLE_LANGS


def _safe_filename(name):
    return re.sub(r'[/\\:*?"<>|]', "-", unicodedata.normalize("NFC", name)).strip()


def title_songs(song_dir, mp3_dir, model="medium", offset=15.0, listen=90.0,
                bitrate_q=0):
    """Name songs from their lyrics and export numbered MP3s.

    Transcribes the opening of each song -- chants are conventionally
    named for their first line -- detects the language, and uses the
    result as a title. Songs whose transcript is not credible (too few
    words, one token repeated, or a language outside TITLE_LANGS, which
    is what non-lexical vocables produce) are left untitled for manual
    naming rather than given an invented name.

    Files are numbered chronologically across the whole recording so they
    sort correctly regardless of title.
    """
    import subprocess
    import tempfile

    import whisper

    files = []
    for sub in sorted(os.listdir(song_dir)):
        d = os.path.join(song_dir, sub)
        if os.path.isdir(d):
            files += sorted(glob.glob(os.path.join(d, "*_song_*.flac")))
    if not files:
        files = sorted(glob.glob(os.path.join(song_dir, "*_song_*.flac")))
    if not files:
        print(f"No songs found under {song_dir}")
        return

    os.makedirs(mp3_dir, exist_ok=True)
    print(f"Naming {len(files)} songs with the '{model}' model...")
    net = whisper.load_model(model)

    manifest = []
    for i, path in enumerate(files, 1):
        y, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=listen)
        tmp = tempfile.mktemp(suffix=".wav")
        sf.write(tmp, y, sr)
        try:
            res = net.transcribe(tmp)
        finally:
            os.remove(tmp)

        text = " ".join(s["text"].strip() for s in res.get("segments", []))
        lang = res.get("language", "")
        title = _clean_title(text) if _titleable(text, lang) else None

        stem = f"{i:02d} - {_safe_filename(title)}" if title else f"{i:02d} - untitled"
        out = os.path.join(mp3_dir, stem + ".mp3")
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path,
               "-codec:a", "libmp3lame", "-q:a", str(bitrate_q),
               "-metadata", f"track={i}",
               "-metadata", f"title={title or os.path.basename(path)[:-5]}",
               "-metadata", "album=Recording", out]
        subprocess.run(cmd, check=True)

        manifest.append({"n": i, "source": path, "language": lang,
                         "title": title, "mp3": out, "transcript": text[:400]})
        print(f"  {i:02d}  {lang or '--':3}  {title or '(untitled -- name manually)'}", flush=True)

    with open(os.path.join(mp3_dir, "titles.json"), "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    named = sum(1 for m in manifest if m["title"])
    print(f"Named {named} of {len(manifest)}; {len(manifest) - named} left untitled -> {mp3_dir}/")


def split_tracks(audio_path, out_dir, silence_db=35, min_silence=3.0,
                 min_track=20.0, pad=0.5, cough_tol=1.0):
    """Split a long recording into individual tracks at quiet gaps.

    The gaps between tracks are rarely true silence -- there are coughs,
    shuffles and ambient noise. So instead of strict silence detection we
    look for *sustained low-energy* regions and tolerate brief loud blips
    (coughs) inside them.

    silence_db   - how many dB below the loud content counts as "quiet"
    min_silence  - minimum sustained quiet length (s) that marks a boundary
    min_track    - drop segments shorter than this (s)
    pad          - padding (s) kept around each track's edges
    cough_tol    - bridge over loud blips (coughs) up to this long (s)
                   so a single cough doesn't break up a quiet gap
    """
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]

    mono = y.mean(axis=0)

    # Frame-level loudness in dB relative to the recording's loud content.
    hop = 512
    frame = 2048
    rms = librosa.feature.rms(y=mono, frame_length=frame, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Reference the "loud" level robustly (95th pct), not the single peak,
    # so a stray transient doesn't skew the threshold.
    loud = np.percentile(rms_db, 95)
    quiet = rms_db < (loud - silence_db)

    sec_per_frame = hop / sr
    cough_frames = int(round(cough_tol / sec_per_frame))
    min_gap_frames = int(round(min_silence / sec_per_frame))

    # Morphological closing: fill short non-quiet holes (coughs) so they
    # don't break an otherwise continuous quiet gap.
    closed = quiet.copy()
    i = 0
    n = len(closed)
    while i < n:
        if not closed[i]:
            j = i
            while j < n and not closed[j]:
                j += 1
            run = j - i
            # Only bridge if flanked by quiet on both sides.
            if run <= cough_frames and i > 0 and j < n and closed[i - 1] and closed[j]:
                closed[i:j] = True
            i = j
        else:
            i += 1

    # Find sustained quiet runs >= min_silence -> these are boundary gaps.
    gaps = []
    i = 0
    while i < n:
        if closed[i]:
            j = i
            while j < n and closed[j]:
                j += 1
            if (j - i) >= min_gap_frames:
                # Cut at the middle of the gap.
                mid = (i + j) // 2
                gaps.append(mid)
            i = j
        else:
            i += 1

    # Build track spans between consecutive gap midpoints.
    cut_samples = [0] + [int(g * hop) for g in gaps] + [y.shape[1]]
    spans = [(cut_samples[k], cut_samples[k + 1]) for k in range(len(cut_samples) - 1)]

    pad_samples = int(pad * sr)
    min_track_samples = int(min_track * sr)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]

    count = 0
    for start, end in spans:
        if end - start < min_track_samples:
            continue
        s = max(0, start - pad_samples)
        e = min(y.shape[1], end + pad_samples)
        track = y[:, s:e]
        if track.shape[0] == 1:
            track = track[0]
        count += 1
        out_path = os.path.join(out_dir, f"{base}_track_{count:02d}.flac")
        _write_audio(out_path, track, sr)
        print(f"  Track {count:02d}: {s / sr:8.1f}s - {e / sr:8.1f}s  ({(e - s) / sr:6.1f}s)  -> {out_path}")

    if count == 0:
        print("No tracks found. Try lowering --silence-db or --min-silence.")
    else:
        print(f"Found {len(gaps)} gaps -> wrote {count} tracks to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Detect and remove unwanted audio events")
    parser.add_argument("input", nargs="?", help="Input audio file")
    parser.add_argument("output", nargs="?", help="Output audio file (required unless --detect-only)")
    parser.add_argument("--events-csv", help="Path to save/load events CSV")
    parser.add_argument("--detect-only", action="store_true", help="Only detect events, don't clean audio")
    parser.add_argument("--use-existing-events", help="Use events from an existing CSV instead of detecting")
    parser.add_argument("--split", action="store_true", help="Split input into individual tracks at silent gaps")
    parser.add_argument("--split-songs", action="store_true",
                        help="Split a concert into individual songs using music vs. applause/speech boundaries")
    parser.add_argument("--title-songs", metavar="SONG_DIR",
                        help="Transcribe songs under SONG_DIR and export numbered, titled MP3s")
    parser.add_argument("--mp3-dir", default="output/mp3_named",
                        help="Where --title-songs writes MP3s (default: output/mp3_named)")
    parser.add_argument("--whisper-model", default="medium",
                        help="Whisper model for --title-songs (default: medium)")
    parser.add_argument("--no-subdivide", action="store_true",
                        help="Do not subdivide a musical passage where the melody or language changes")
    parser.add_argument("--music-only", action="store_true",
                        help="Write only musical passages, skipping the spoken sections between them")
    parser.add_argument("--min-song", type=float, default=45.0,
                        help="Drop songs shorter than this many seconds (default: 45)")
    parser.add_argument("--out-dir", default="tracks", help="Output directory for --split (default: tracks)")
    parser.add_argument("--silence-db", type=float, default=35, help="dB below loud content treated as quiet (default: 35)")
    parser.add_argument("--min-silence", type=float, default=3.0, help="Min sustained quiet in seconds marking a track boundary (default: 3.0)")
    parser.add_argument("--min-track", type=float, default=20.0, help="Drop tracks shorter than this many seconds (default: 20)")
    parser.add_argument("--cough-tol", type=float, default=1.0, help="Bridge over loud blips (coughs) up to this long in seconds (default: 1.0)")
    parser.add_argument("--sensitivity", choices=sorted(SENSITIVITY_PRESETS), default="strict",
                        help="Detection sensitivity preset; looser flags more events (default: strict)")
    parser.add_argument("--multi-pass", action="store_true",
                        help="Run detection at every sensitivity level and save one events CSV per level for comparison")
    parser.add_argument("--method", choices=["inpaint", "stem", "cut"], default="inpaint",
                        help="How to clean events: 'inpaint' removes only the noise energy and keeps music/voice "
                             "underneath (highest fidelity, no level dip); 'stem' swaps in Demucs music-only audio; "
                             "'cut' splices them out (default: inpaint)")
    parser.add_argument("--detector", choices=["panns", "spectral"], default="panns",
                        help="Event detector: 'panns' recognises cough/throat-clearing/sneeze via a pretrained "
                             "AudioSet model; 'spectral' uses the loudness/noisiness heuristic (default: panns)")
    parser.add_argument("--keep-lossy", action="store_true",
                        help="Honour a .mp3/.ogg output path instead of redirecting to FLAC. "
                             "Only for final delivery -- it re-encodes the whole recording")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="PANNs class probability needed to flag an event (default: 0.3; lower finds more). "
                             "True coughs score ~0.5-0.7; music sits below ~0.1")
    args = parser.parse_args()

    if args.multi_pass:
        base = os.path.splitext(os.path.basename(args.input))[0]
        print(f"Multi-pass detection on {args.input}...")
        results = {}
        for level in ["strict", "medium", "loose"]:
            events = detect_events(args.input, sensitivity=level)
            csv_path = f"{base}_events_{level}.csv"
            save_events_csv(events, csv_path)
            total = sum(e - s for s, e in events)
            results[level] = (len(events), total, csv_path)
            print(f"  {level:>6}: {len(events):3d} events, {total:7.1f}s flagged -> {csv_path}")
        print("\nCompare the CSVs (or spot-check clips), then clean with the level you like:")
        best = results["strict"][2]
        print(f"  python clean_audio.py '{args.input}' output.mp3 --use-existing-events {best}")
        return

    if args.title_songs:
        title_songs(args.title_songs, args.mp3_dir, model=args.whisper_model)
        return

    if args.split_songs:
        print(f"Splitting {args.input} into songs...")
        split_songs(args.input, args.out_dir, min_song=args.min_song,
                    music_only=args.music_only, subdivide=not args.no_subdivide)
        return

    if args.split:
        print(f"Splitting {args.input} into tracks...")
        split_tracks(
            args.input,
            args.out_dir,
            silence_db=args.silence_db,
            min_silence=args.min_silence,
            min_track=args.min_track,
            cough_tol=args.cough_tol,
        )
        return

    if not args.detect_only and not args.output:
        parser.error("output path is required unless --detect-only is set")

    if args.use_existing_events:
        print(f"Loading events from {args.use_existing_events}")
        events = load_events_csv(args.use_existing_events)
        print(f"Loaded {len(events)} events")
    else:
        if args.detector == "panns":
            print(f"Detecting events in {args.input} (PANNs, threshold {args.threshold})...")
            events = detect_events_panns(args.input, threshold=args.threshold)
        else:
            print(f"Detecting events in {args.input} (spectral, sensitivity: {args.sensitivity})...")
            events = detect_events(args.input, sensitivity=args.sensitivity)
        print(f"Detected {len(events)} events")

    if args.events_csv:
        save_events_csv(events, args.events_csv)
        print(f"Events saved to {args.events_csv}")

    if args.detect_only:
        for i, (s, e) in enumerate(events):
            print(f"  Event {i:03d}: {s:.2f}s - {e:.2f}s (duration: {e - s:.2f}s)")
        return

    out_path = args.output if args.keep_lossy else _lossless_path(args.output)
    if out_path != args.output:
        print(f"Writing lossless FLAC instead of {os.path.splitext(args.output)[1]} "
              f"(use --keep-lossy to override)")
    args.output = out_path

    print(f"Cleaning audio -> {args.output} (method: {args.method})")
    if args.method == "inpaint":
        inpaint_events(args.input, args.output, events)
    elif args.method == "stem":
        repair_events(args.input, args.output, events)
    else:
        remove_events(args.input, args.output, events)
    print("Done.")


if __name__ == "__main__":
    main()
