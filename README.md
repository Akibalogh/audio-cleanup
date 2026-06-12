# audio-cleanup

Clean up long audio recordings: detect and remove unwanted audio events
(coughs, retches, other transient noises), and split a single multi-hour
recording into individual tracks.

Everything lives in one script: [`clean_audio.py`](clean_audio.py).

## Requirements

- Python 3.9+

```bash
pip install -r requirements.txt
```

`librosa`/`numpy`/`soundfile` cover detection, cutting, and splitting.
`demucs`/`torch` are needed for the default stem-swap cleaning method.

## Usage

### Detect and clean events (one step)

```bash
python clean_audio.py input.mp3 output.mp3
```

By default events are **repaired, not cut**: Demucs separates each event
window into vocals vs. accompaniment, and the event range is replaced
with the accompaniment-only audio. The music plays through uninterrupted,
the recording's duration is unchanged, and audio outside the event
windows is untouched. Use `--method cut` to splice events out instead
(shorter output, loses the music under each event).

### Detect only (review before cleaning)

```bash
python clean_audio.py input.mp3 --detect-only --events-csv events.csv
```

Prints each detected event with timestamps and saves them to a CSV
(`start_seconds,end_seconds` per line). Review/edit the CSV, then clean:

```bash
python clean_audio.py input.mp3 output.mp3 --use-existing-events events.csv
```

### Tune detection sensitivity

Detection has three presets — `strict` (default), `medium`, and `loose`.
Looser settings flag more events (at the risk of more false positives).

```bash
python clean_audio.py input.mp3 output.mp3 --sensitivity medium
```

To compare all levels in one run, use multi-pass. It detects at every
sensitivity level and writes one events CSV per level — no audio is
modified:

```bash
python clean_audio.py input.mp3 --multi-pass
#   strict:  24 events,   92.3s flagged -> input_events_strict.csv
#   medium:  31 events,  118.7s flagged -> input_events_medium.csv
#   loose :  40 events,  151.2s flagged -> input_events_loose.csv
```

Pick the level whose CSV looks right, then clean with it via
`--use-existing-events`.

### Split a long recording into tracks

Splits at sustained quiet gaps. The gaps don't need to be true silence —
brief loud blips (a cough, a chair scrape) inside a gap are bridged over.

```bash
python clean_audio.py recording.mp3 --split --out-dir tracks
```

Tuning flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--silence-db` | 35 | How many dB below the loud content counts as "quiet". Lower = stricter (gap must be quieter). |
| `--min-silence` | 3.0 | Minimum sustained quiet (seconds) to mark a track boundary. |
| `--min-track` | 20 | Drop segments shorter than this (seconds). |
| `--cough-tol` | 1.0 | Bridge over loud blips up to this long (seconds) inside a quiet gap. |
| `--out-dir` | `tracks` | Output directory. |

If it over-splits (cuts inside tracks), lower `--silence-db` or raise
`--min-silence`. If it misses boundaries because of noises in the gaps,
raise `--cough-tol`.

## How it works

**Event detection** computes frame-level RMS energy, spectral flatness,
zero-crossing rate, and spectral rolloff, then flags frames where all
four exceed percentile thresholds (the sensitivity preset). Flagged runs
of 0.4–10s become events, padded slightly and merged when close together.

**Cleaning** has two methods. `stem` (default): each event window plus a
few seconds of context is separated by Demucs (htdemucs, two-stem); the
event range is patched with the accompaniment-only stem, crossfaded at
the edges, so no music is lost and timing is preserved. `cut`: the
flagged ranges are spliced out with short crossfades.

**Track splitting** measures frame loudness relative to the recording's
loud content (95th percentile), finds sustained quiet runs, bridges short
loud blips inside them, and cuts at each gap's midpoint.

## Typical workflow for a long recording

```bash
# 1. Compare detection sensitivities
python clean_audio.py recording.mp3 --multi-pass

# 2. Clean with the level you picked
python clean_audio.py recording.mp3 cleaned.mp3 --use-existing-events recording_events_strict.csv

# 3. Split the cleaned file into tracks
python clean_audio.py cleaned.mp3 --split --out-dir tracks
```
