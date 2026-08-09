# audio-cleanup

Clean up long recordings: remove coughs and similar transient noises
without damaging the material underneath, and split a multi-hour file
into individual songs.

Everything lives in one script: [`clean_audio.py`](clean_audio.py).

## Requirements

- Python 3.9+

```bash
pip install -r requirements.txt
```

`librosa`/`numpy`/`soundfile` are the core. `panns-inference`/`torch`
power the default detector and the song splitter; the model checkpoint
(~320MB) downloads automatically on first use. `demucs` is only needed
for the optional `--method stem`.

## Usage

### Clean a recording

```bash
python clean_audio.py raw/recording.mp3 output/cleaned.flac
```

Detects coughs / throat clearing / sneezes and repairs them in place.
Duration is unchanged and audio outside the detected events is left
bit-exact.

### Detect only (review before cleaning)

```bash
python clean_audio.py raw/recording.mp3 --detect-only --events-csv events.csv
```

Writes `start_seconds,end_seconds` per line. Review or edit the CSV,
then clean using it:

```bash
python clean_audio.py raw/recording.mp3 output/cleaned.flac --use-existing-events events.csv
```

### Split into individual songs

Live recordings separate songs with applause and talking rather than
silence, so this scores each second for music vs. applause/speech,
smooths over 20s, and cuts at sustained music↔non-music transitions.

```bash
python clean_audio.py output/cleaned.flac --split-songs --out-dir output/songs
```

`--min-song` (default 45s) discards fragments. Raise it if it
over-splits; lower it if two songs get merged.

## Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--detector` | `panns` | `panns` recognises actual sound classes; `spectral` is the old loudness heuristic (kept as a fast fallback, misses most events) |
| `--threshold` | `0.3` | Class probability to flag an event. Real coughs score 0.5–0.7, music sits below 0.1 |
| `--method` | `inpaint` | `inpaint` removes only the noise energy; `stem` swaps in Demucs accompaniment; `cut` splices the range out |
| `--keep-lossy` | off | Honour an `.mp3` output path instead of redirecting to FLAC |

Detection is deliberately conservative. Because a false positive still
costs a repair, prefer raising `--threshold` over lowering it unless
events are being missed.

## How it works

**Detection** runs PANNs (an AudioSet-pretrained CNN14 tagger) over 1s
windows at 0.5s hop and flags windows where Cough / Throat clearing /
Sneeze exceeds the threshold. Hysteresis extends each event while
neighbouring windows stay above a third of the threshold, so events
aren't truncated mid-cough.

**Repair** (`inpaint`) first narrows the ~1–4s flagged window down to
the actual transient using a 10ms energy envelope — a cough is ~0.4s, so
repairing the whole window would needlessly process good audio. It then
takes a median reference spectrum from clean guard regions either side
(excluding any neighbouring event), interpolates that reference across
the event so evolving music is tracked, and attenuates each
time-frequency bin down to the expected level. Phase is preserved and
one shared gain mask is applied to both channels, so the stereo image
doesn't shift.

Median rather than max matters: a max-based reference is set by any
transient in the guard region and leaves the cough untouched. There is
deliberately no gain floor — a cough sits 25–35dB above quiet ambience,
so a floor would only make it quieter rather than remove it.

**Song splitting** scores Music against Applause/Speech/Cheering, smooths
both over 20s, drops runs too short to be a real passage, merges
same-label neighbours, and cuts at the remaining transitions.

**Output is 24-bit FLAC.** The repair touches well under 1% of a
recording, so encoding the result to MP3 would impose generation loss on
all of it — and splitting afterwards would encode a second time. An
`.mp3` output path is redirected to `.flac` unless `--keep-lossy` is
given.

## Folder layout

```
raw/        original recordings (gitignored)
events/     detected event CSVs (gitignored)
output/     cleaned audio and split songs (gitignored)
```

## Typical workflow

```bash
# 1. Clean each chunk (writes lossless FLAC + an event CSV to review)
python clean_audio.py raw/chunk_0.mp3 output/clean/chunk_0_clean.flac \
    --events-csv events/chunk_0.csv

# 2. Split the cleaned audio into songs
python clean_audio.py output/clean/chunk_0_clean.flac \
    --split-songs --out-dir output/songs/chunk_0
```
