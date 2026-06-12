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


def main():
    parser = argparse.ArgumentParser(description="Detect and remove unwanted audio events")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", nargs="?", help="Output audio file (required unless --detect-only)")
    parser.add_argument("--events-csv", help="Path to save/load events CSV")
    parser.add_argument("--detect-only", action="store_true", help="Only detect events, don't clean audio")
    parser.add_argument("--use-existing-events", help="Use events from an existing CSV instead of detecting")
    args = parser.parse_args()

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
