# Delivery verification

Repository: https://github.com/DavidMetts/multimodal-preprocessing (private).

Source and nine-event synthetic demo are separate from the raw generator.
Entry point: `app.py`; branch: `main`.

Local verification on 2026-09-19: all seven unittest cases passed, including Streamlit AppTest startup and reference controls. Three real fixture tone means were 220.0000188, 440.0000073, and 880.0009253 Hz. Maximum per-second variances were 1.71e-8, 2.12e-7, and 1.90e-6 Hz². Selected image areas were 14,313, 26,077, and 37,841 pixels for one, two, and three scoops, equal across flavors. Shared demo vocabulary contains 52 terms.

Public dashboard: https://davidmetts-multimodal-preprocessing.streamlit.app/

Hosted verification: app renders its Text, Image and Audio tabs; the actual raw demo ZIP was uploaded through the hosted file picker and processed as nine events; the combined export produced a browser download. Image and spectrogram/pitch layouts were visually inspected. Playback remained user-initiated. A fresh anonymous HTTP cookie session returned HTTP 200 at the app URL without a sign-in redirect. The Streamlit public-sharing switch is enabled; GitHub source remains private.

Known scope limits: image mask is specific to current generator artwork; audio estimator assumes clean single tones; missing/invalid measurements remain blank. No fusion, clustering, classification, prediction, or training is included.
