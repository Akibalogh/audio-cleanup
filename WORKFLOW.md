# Audio cleanup workflow

Removing coughs from a long recording and splitting it into songs.

This replaces an earlier workflow built on Demucs stem separation and a
spectral event detector. Both were measured against what replaced them on a
real 4.9-hour recording and lost. The comparisons are recorded at the end so
the old approach is not rebuilt by mistake.

# What it does

- Finds coughs, throat clearing and sneezes using a pretrained sound classifier.
- Repairs each one in place, leaving music and singing underneath at full level.
- Splits the recording into individual songs, including songs that run back to back.
- Writes lossless masters, then MP3s for listening.

# What it does not do

- It does not remove every unwanted sound. Detection is deliberately conservative, because a false positive costs a repair on good audio.
- It does not name songs. Transcription was tried and produced unusable titles: much of the singing is non-lexical, so the model reported implausible languages and returned syllables.
- It does not get every song boundary right. Two needed correcting by ear.

# Prerequisites

```
cd ~/apps/audio-cleanup
source audio-cleanup-venv/bin/activate
pip install -r requirements.txt

# ffmpeg is also needed, for the MP3 export step
brew install ffmpeg
```

The classifier checkpoint (about 320 MB) downloads automatically on first use.

# Folder layout

```
~/apps/audio-cleanup/
  clean_audio.py          the whole pipeline, one script
  raw/                    original recordings
  events/                 detected cough timestamps (CSV)
  output/
    clean/                repaired full-length masters (FLAC)
    songs/                per-song masters (FLAC)
    songs_numbered/       01.mp3 ... NN.mp3 plus index.txt
    samples/              before/after clips for spot checking
```

# Step 1: split the recording into chunks

A five-hour file needs more memory than most machines have once decoded.
Three chunks of roughly 98 minutes each work well.

```
ffmpeg -y -i raw/recording.mp3 -f segment -segment_time 5880 \
  -c copy raw/chunk_%d.mp3
```

# Step 2: detect and repair

Run per chunk. This writes a lossless master and a CSV of every timestamp it
repaired.

```
for c in chunk_0 chunk_1 chunk_2; do
  python clean_audio.py "raw/$c.mp3" "output/clean/${c}_clean.flac" \
      --events-csv "events/${c}.csv"
done
```

On the reference recording this found 110 events across 4.9 hours, about one
every 2.5 minutes, taking roughly 5 minutes per chunk on CPU.

# Step 3: check the repairs by ear

This is the step that matters most and the one no measurement replaces. Pull
the largest repairs from the event CSVs and listen before going further.

An early version of the repair measured as working when it was in fact barely
attenuating the coughs. Only listening caught it.

# Step 4: split into songs

```
for c in chunk_0 chunk_1 chunk_2; do
  python clean_audio.py "output/clean/${c}_clean.flac" \
      --split-songs --out-dir "output/songs/$c"
done
```

Songs are separated by talking rather than silence, so the split looks for
music-to-speech transitions, then subdivides any passage where the melody or
language changes. Spoken passages are written separately as `talk_NN` so they
are not mistaken for songs.

Expect to correct a boundary or two by ear. Merged songs and over-split songs
both happen, and the automated confidence scores did not predict either
reliably.

# Step 5: export numbered MP3s

Number songs chronologically across the whole recording so they sort
correctly, and keep an index mapping each number back to its position in the
original.

```
ffmpeg -nostdin -y -i "$song" -codec:a libmp3lame -q:a 0 \
  -metadata track="$n" "output/songs_numbered/$(printf %02d $n).mp3"
```

# Key options

| Flag | Default | What it controls |
|------|---------|------------------|
| `--threshold` | 0.3 | Confidence needed to flag a cough. Real coughs score 0.5-0.7; music sits below 0.1. Raise rather than lower it. |
| `--method` | inpaint | `inpaint` removes only the noise energy; `cut` splices the range out and shortens the recording. |
| `--min-song` | 45 | Passages shorter than this are discarded. |
| `--no-subdivide` | off | Keep a long passage whole instead of splitting where the melody changes. |
| `--keep-lossy` | off | Allow an MP3 output path instead of redirecting to FLAC. |

# Why the repair works this way

A cough is removed by attenuating only the time-frequency cells where it
exceeds what the surrounding music reaches, so singing that continues through
the cough keeps its level. Three details decide whether this works at all:

- The reference spectrum is the **median** of the clean audio either side, not the maximum. A maximum is set by any transient in that window, which leaves the cough untouched.
- There is **no floor** on the attenuation. A cough sits 25-35 dB above quiet ambience, so a floor only makes it quieter instead of removing it.
- The flagged window is **narrowed to the actual burst** first. The classifier reports 1-4 seconds; a cough lasts about 0.4.

Measured on the reference recording: 10-36 dB of attenuation on the cough,
with audio outside each event bit-exact and duration unchanged.

# Fixing a track's balance

If one song has an instrument drowning the vocals, separate it into stems and
**subtract** the offending stem from the original rather than rebuilding the
track from the stem sum. Summing stems makes the whole track a separation
artifact, which sounds washed out.

```
python -m demucs -n htdemucs -o /tmp/rebal "output/songs/chunk_2/song.flac"
# then: out = original + (10**(gain_db/20) - 1) * stem
```

Note the trade-off: if the loud instrument carries much of the track's energy,
removing it makes the track quieter, and a peak-normalised source leaves no
headroom to make that back. Reducing by 6 dB rather than 9 usually keeps more
punch while still freeing the vocals.

# Approaches tried and rejected

| Approach | Why it was dropped |
|----------|--------------------|
| Demucs stem swap | Replacing the event with the accompaniment stem drops 16-20 dB, because the vocal stem contains the singing itself. |
| Spectral heuristic | Requiring loudness, flatness, zero-crossing and rolloff to peak together found 1 event per 98 minutes where the classifier found 41. |
| Silence-based splitting | Songs here are separated by talking, not silence, so the real boundaries were never found. |
| Lyric transcription | Much of the singing is non-lexical; the model returned syllables and implausible languages. |

# A note on verification

Every quality claim above came from measuring one approach against another on
the real recording, not from reasoning about what should work. Several
confident-looking metrics turned out to be wrong, including one that ranked a
merged track as a single song and a single song as merged. Where a measurement
and a listener disagreed, the listener was right every time.

---

To regenerate the Word version:

```
pandoc WORKFLOW.md -o workflow_notes.docx --toc --toc-depth=1
```
