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
import os
import sys
import tempfile

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
        if segment.shape[1] > fade_samples:
            fade_out = np.linspace(1, 0, fade_samples)
            segment[:, -fade_samples:] *= fade_out
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

    sf.write(output_path, cleaned.T if cleaned.ndim > 1 else cleaned, sr)

    original_dur = y.shape[1] / sr
    cleaned_dur = cleaned.shape[-1] / sr
    removed_dur = original_dur - cleaned_dur
    print(f"Original: {original_dur:.1f}s | Cleaned: {cleaned_dur:.1f}s | Removed: {removed_dur:.1f}s ({len(events)} events)")


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
    sf.write(output_path, out.T if out.ndim > 1 else out, sr)
    total = sum(e - s for s, e in events)
    print(f"Repaired {len(events)} events ({total:.1f}s) in place -- duration unchanged.")


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
        out_path = os.path.join(out_dir, f"{base}_track_{count:02d}.mp3")
        sf.write(out_path, track.T if track.ndim > 1 else track, sr)
        print(f"  Track {count:02d}: {s / sr:8.1f}s - {e / sr:8.1f}s  ({(e - s) / sr:6.1f}s)  -> {out_path}")

    if count == 0:
        print("No tracks found. Try lowering --silence-db or --min-silence.")
    else:
        print(f"Found {len(gaps)} gaps -> wrote {count} tracks to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Detect and remove unwanted audio events")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", nargs="?", help="Output audio file (required unless --detect-only)")
    parser.add_argument("--events-csv", help="Path to save/load events CSV")
    parser.add_argument("--detect-only", action="store_true", help="Only detect events, don't clean audio")
    parser.add_argument("--use-existing-events", help="Use events from an existing CSV instead of detecting")
    parser.add_argument("--split", action="store_true", help="Split input into individual tracks at silent gaps")
    parser.add_argument("--out-dir", default="tracks", help="Output directory for --split (default: tracks)")
    parser.add_argument("--silence-db", type=float, default=35, help="dB below loud content treated as quiet (default: 35)")
    parser.add_argument("--min-silence", type=float, default=3.0, help="Min sustained quiet in seconds marking a track boundary (default: 3.0)")
    parser.add_argument("--min-track", type=float, default=20.0, help="Drop tracks shorter than this many seconds (default: 20)")
    parser.add_argument("--cough-tol", type=float, default=1.0, help="Bridge over loud blips (coughs) up to this long in seconds (default: 1.0)")
    parser.add_argument("--sensitivity", choices=sorted(SENSITIVITY_PRESETS), default="strict",
                        help="Detection sensitivity preset; looser flags more events (default: strict)")
    parser.add_argument("--multi-pass", action="store_true",
                        help="Run detection at every sensitivity level and save one events CSV per level for comparison")
    parser.add_argument("--method", choices=["stem", "cut"], default="stem",
                        help="How to clean events: 'stem' replaces them with music-only audio via Demucs "
                             "(no music lost, duration unchanged); 'cut' splices them out (default: stem)")
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
        print(f"Detecting events in {args.input} (sensitivity: {args.sensitivity})...")
        events = detect_events(args.input, sensitivity=args.sensitivity)
        print(f"Detected {len(events)} events")

    if args.events_csv:
        save_events_csv(events, args.events_csv)
        print(f"Events saved to {args.events_csv}")

    if args.detect_only:
        for i, (s, e) in enumerate(events):
            print(f"  Event {i:03d}: {s:.2f}s - {e:.2f}s (duration: {e - s:.2f}s)")
        return

    print(f"Cleaning audio -> {args.output} (method: {args.method})")
    if args.method == "stem":
        repair_events(args.input, args.output, events)
    else:
        remove_events(args.input, args.output, events)
    print("Done.")


if __name__ == "__main__":
    main()
