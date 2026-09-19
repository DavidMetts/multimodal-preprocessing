# Delivery verification

Repository: https://github.com/DavidMetts/multimodal-preprocessing (private).

Source and nine-event synthetic demo are separate from the raw generator.
Entry point: `app.py`; branch: `main`.

Local verification on 2026-09-19: all seven unittest cases passed, including Streamlit AppTest startup and reference controls. Three real fixture tone means were 220.0000188, 440.0000073, and 880.0009253 Hz. Maximum per-second variances were 1.71e-8, 2.12e-7, and 1.90e-6 Hz². Selected image areas were 14,313, 26,077, and 37,841 pixels for one, two, and three scoops, equal across flavors. Shared demo vocabulary contains 52 terms.

The public app link and final browser verification will be recorded after hosting completes.
