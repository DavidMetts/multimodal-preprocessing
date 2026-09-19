# Multimodal Preprocessing

A small, inspectable Streamlit dashboard for the **second stage** of a controlled synthetic-data project. The separate generator produces the raw sources; this app measures those files. It does not train models, classify, cluster, fuse modalities, or assign feature importance.

## Start locally

Use Python 3.11–3.13. From this project folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Open the local address printed by Streamlit. It is only a local address, not the hosted public link. The nine-event demo loads immediately; choose an event and click Text, Image, or Audio. Audio playback starts only when you press Play.

## Import your raw sources

The simplest route is **Event ZIP uploads**. Upload several generator event ZIPs, or one ZIP containing multiple event folders. Each event folder uses these exact names:

```text
event_id/
  manifest.json
  text.txt
  image.png
  audio.wav
```

The manifest supplies `event_id` and `event_time`; the other fields are evaluation context and are never feature inputs. Missing modalities remain missing. Invalid files show an explanation in Collection quality and in the CSV status/error columns. Duplicate event IDs are rejected rather than silently overwritten. Empty/invalid text is not used to fit a reference.

For separate downloads, choose **Single event files**, upload any available modalities and the optional manifest, and click **Add event to collection**. Repeat to collect several events. Without a manifest, the app creates a content-derived `unlinked_…` ID; it cannot recover the original event identity or time. Repeated identical unlinked events require explicit distinct manifest IDs. The original filenames do not determine attributes or class labels.

A hosted app cannot read a folder on your computer. Zip your event folders, or run this local helper:

```powershell
python package_folder.py "C:\your\event-folder" collection.zip
```

The helper refuses to overwrite an existing output. Upload the resulting ZIP. Upload limits: 50 events, 25 MB combined compressed/expanded bytes, 250 archive entries per ZIP, and 2 MB per raw file. Only the four canonical names are accepted. Unsafe paths, duplicate ZIP members, and archive links are rejected. ZIP content is read into the current session's memory, never extracted into arbitrary paths. Uploads are not stored in a shared server folder or global cache. Keep your own raw downloads: browser sessions and cloud storage are not permanent archives.

## The three transformations

### Text: individual words and shared TF-IDF

Tokens are lowercase matches of `[a-z]+(?:'[a-z]+)?`. Internal apostrophes are retained; punctuation and numbers are ignored. There is no stop-word removal, stemming, sentiment, flavor weighting, or fixed twenty-word target. Vanilla, chocolate, and strawberry remain ordinary words. All reference terms are retained up to a 2,000-term safety limit; an oversized vocabulary is rejected, not silently trimmed.

The first valid collection automatically establishes a reference. Select reference documents and click **Fit and freeze reference** to change it intentionally. Merely uploading later events does not refit it. Reference provenance lists event IDs and text SHA-256 hashes. Restore the downloaded `text_reference.json` in a new session to apply exactly the same vocabulary order and IDF.

For term t in document d:

- Count = occurrences of t.
- TF = count / all document tokens, including out-of-vocabulary tokens.
- DF = number of reference documents containing t.
- IDF = `ln((1 + N)/(1 + DF)) + 1` for N reference documents.
- TF-IDF = TF × IDF.
- Normalized coordinate = TF-IDF divided by the L2 length across the frozen vocabulary. A zero vector remains zero and unknown-token count is reported.

With one reference document, all observed terms have IDF 1: cross-document distinctiveness cannot yet be established. The word chart shows actual words in the selected document, including terms outside a restored vocabulary. The table and heatmap use the frozen vocabulary. The heatmap shows raw TF-IDF, not labels.

### Image: four attributes

For the current **512 × 640 PNG** artwork, select pixels inside x=189…323 and y=61…387, then exclude the exact background RGB (229,231,235), cone (199,138,69), and outline (156,101,47). This geometric/color mask is shared across every event; it never selects a mask based on flavor. The highlighted preview dims excluded pixels.

The four candidates are mean red, green, blue of selected pixels, and selected area / full canvas area. Raw RGB is 0…255; normalized RGB is raw / 255. Area fraction is already in [0,1]. Selected and canvas pixel counts are retained as inspection measurements, not extra candidate coordinates. More scoops increase selected area while a fixed flavor's average color stays constant. At equal scoop count all three colors have the same area.

This is intentionally **not a general object segmenter**. Changed colors for the cone/background, JPEG artifacts, shading, resizing, extra objects, or revised artwork require a revised mask and new version. The size check alone does not prove arbitrary uploaded images match the controlled scene. No edge attributes are included.

### Audio: twenty ordered attributes

Input is exactly ten seconds of mono, uncompressed 16-bit PCM WAV at 8,000–48,000 samples/second (the current generator uses 16,000). Read the waveform, split into 100 non-overlapping 100 ms frames, and estimate pitch from the mean interval between rising zero crossings with linear interpolation. This estimator is suitable for the restricted clean, steady, single-sine baseline, not general speech/music or harmonic mixtures. It is not a proof that arbitrary uploaded audio contains only one tone.

A frame needs RMS ≥ 0.0001, at least three crossings, period coefficient of variation ≤ 0.05, and measured pitch 100…1200 Hz. Otherwise it is silent/invalid. All ten valid frames are required for a one-second summary; invalid windows get blank numeric values, never fabricated zero pitch. The ten chronological sections are [0,1), [1,2), … [9,10) seconds.

Each section contributes **mean pitch (Hz)** then **population pitch variance (Hz²)**: twenty coordinates in chronological order. All raw values are preserved. Means normalize by dividing by a fixed 1,000 Hz; variances divide by fixed 1,000,000 Hz². These are stable unit scales, not per-clip min/max scaling; they preserve tone differences and do not amplify tiny nearly constant variances. No clipping occurs, so a valid 1,200 Hz tone normalizes to 1.2. The scales are not learned feature importance.

The spectrogram visualizes spectral power density in dB. Pitch is estimated independently from waveform crossings. Per-second summaries lose within-second order; the bundle also retains timestamped frame measurements. The variance chart uses a fixed 0–1 Hz² reference range for these steady tones; the numeric table is authoritative for other valid inputs.

## Export and replay

Download three CSV tables individually or **All tables + metadata**. The bundle contains:

| File | Contents |
|---|---|
| `text_attributes.csv` | Counts, TF, raw TF-IDF and normalized TF-IDF for each frozen term; total/unknown token counts |
| `image_attributes.csv` | Raw RGB and pixel counts/area; four normalized candidates |
| `audio_attributes.csv` | Twenty raw attributes (Hz, Hz²), followed by twenty scaled attributes |
| `metadata.json` | Settings, units, candidate feature order, source hashes, event times, normalization rules and runtime versions |
| `text_reference.json` | Frozen vocabulary, DF, IDF, fit collection and tokenization settings |
| `inspection/audio_frames.csv` | Validity, RMS, measured pitch and start/center/end times for every frame |

Every modality table has one row for every event, including missing/invalid modalities. Link with `event_id`; trace to the untouched raw file with `source_reference` and `source_sha256`. Blank numeric cells mean missing/invalid; do not turn them into zeros. IDF and DF are shared reference metadata rather than duplicate event coordinates. Candidate column order is explicitly recorded separately from raw measurement columns. Unequal modality sizes are accepted and imply neither equivalent information nor importance.

To replay, retain the raw sources, this source version and the export metadata. Load the same raw files, restore `text_reference.json`, and export again. Image and audio use fixed settings recorded in metadata. Changing a reference is a new transformation; keep its metadata with its tables. No fusion table is generated here.

## Files and the next change

`app.py` owns the interface. `preprocessing/inputs.py` validates collections; `text.py`, `image.py`, and `audio.py` perform their respective transformations without Streamlit; `export.py` assembles the tables and metadata. `package_folder.py` is the local import helper. `demo_collection.zip` holds unchanged copies of the original generator's assets, assembled into nine fixture events without original event timestamps. Only `tests/fixture_truth.json` contains evaluation labels; feature extraction never imports it.

Make the next algorithm change in one modality module, adjust its metadata/version, add an invariant test, and run:

```powershell
python -m unittest discover -s tests -v
```

Tests cover vocabulary consistency, exact colors and increasing area, measured tones, chronological ordering, silence, metadata-label independence, missing/invalid sources, safe ZIP paths, aligned exports, replay, and Streamlit startup/reference controls.

## GitHub and Streamlit publication

Use the separate `DavidMetts/multimodal-preprocessing` repository. Preserve its existing history; commit the project contents at the repository root with `app.py` and `requirements.txt` visible there. Keep source private as requested. Do not upload a nested parent folder or commit local uploaded data/secrets. For browser uploads, expand the project ZIP first; uploading the project ZIP alone does not deploy the app.

In Streamlit Community Cloud, choose Create app → deploy from GitHub, choose this repository, branch `main`, and main file `app.py`. Choose Python 3.13 in Advanced settings if needed. After deployment, public app sharing is configured independently from GitHub repository privacy. Later code updates go through a normal commit/push to this repository's deployed branch; Streamlit rebuilds the app. The raw generator remains a separate repository and app.

See `DEPLOYMENT.md` for the actual verified repository/app links and verification results for this delivery.
