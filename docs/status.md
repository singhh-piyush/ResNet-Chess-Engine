# Release status — 2026-10-09

`candidate-v2` is live on the free CPU Hugging Face Space, deployed from GitHub `main` through the scoped trusted publisher. It is served as ONNX with the new search and an opening book.

- **Model:** the 15-block network fine-tuned from the legacy fold-0 checkpoint on the 4272-action vocabulary. `legacy-cv0` stays on the Space; `models/legacy-cv0.json` is the rollback manifest.
- **Search:** 2-second soft budget and 3-second hard cap, replacing the fixed 9.5 seconds. Book moves come from 49 positions in the last 18 months of train and validation games.
- **Hosted latency:** 10 fresh searches on the Space took 0.65–1.50 s on the server, 1.4–2.2 s including the network round trip. Book moves answer in about 1 ms.
- **Quality gate:** passed on 1,600 positions from the 69 test games that no model has trained on. Blunders fell from 18.0% to 16.1% (95% CI of the change −3.3 to −0.6 points). Style top-1 went from 40.1% to 40.5% (CI −1.0 to +1.8). p95 latency on two CPU threads was 2.43 s.
- **Evaluation caveat:** `legacy-cv0` trained on 262 of the 331 frozen test games, because `GroupKFold` tie-breaking differs from the original run. Earlier figures near 73% measured memorization. See the README.
- **K-fold and distillation:** all eight fine-tuning configs tied within noise. A from-scratch 10x128 student reached 39.8% top-1, so distillation was paused. Further style gains likely need more data, such as rating-matched Lichess games.

Validation: 22 backend and training checks passed. The interface builds, and lint is clean. Chrome confirmed that the live landing page shows the updated stats and that the live engine plays book moves. Phone layouts were checked at 390 and 360 px.
