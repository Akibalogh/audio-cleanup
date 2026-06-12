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


def detect_events(audio_path, sr=16000):
    """Detect unwanted audio events using spectral analysis."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    frame = 4096
    hop = 512

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, roll_percent=0.85)[0]

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    score = (
        (rms_db > np.percentile(rms_db, 86))
        & (flat > np.percentile(flat, 72))
        & (zcr > np.percentile(zcr, 70))
        & (roll > np.percentile(roll, 62))
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
    args = parser.parse_args()

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
        print(f"Detecting events in {args.input}...")
        events = detect_events(args.input)
        print(f"Detected {len(events)} events")

    if args.events_csv:
        save_events_csv(events, args.events_csv)
        print(f"Events saved to {args.events_csv}")

    if args.detect_only:
        for i, (s, e) in enumerate(events):
            print(f"  Event {i:03d}: {s:.2f}s - {e:.2f}s (duration: {e - s:.2f}s)")
        return

    print(f"Cleaning audio -> {args.output}")
    remove_events(args.input, args.output, events)
    print("Done.")


if __name__ == "__main__":
    main()
