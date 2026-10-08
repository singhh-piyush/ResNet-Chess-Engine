# Backend repair status — 2026-10-09

The cleaned backend is deployed from GitHub main to the existing free CPU Hugging Face Space. A scoped GitHub trusted publisher was configured through Chrome; no persistent HF token was added to GitHub.

- Legacy source, weights, datasets, logs, generated bundles and unused starter assets are preserved under ignored `archive/`, with a checksum manifest.
- Shared promotion-safe encoding, validated API, bounded neural search and real progress streaming are implemented.
- The current release is the previous fold-0 baseline. It is not claimed to be a newly improved model.
- Refreshed public archive: 3,457 games, including 3,451 unique standard games. Deterministic relabeling completed with zero errors: 206,088 positions.
- Frozen split: 164,581 training, 20,934 validation, 20,573 test positions. Style labels: 67,190 / 8,411 / 8,389 respectively.
- GPU training launched on the local RTX 5060 with batches of 256, early stopping and resumable checkpoints. Independent evaluation runs afterward; candidate promotion remains conditional on the quality gate and hosted verification.
- Progress monitoring is configured in this chat. Training logs and checkpoints remain ignored locally.

Validation: 12 backend/training checks passed. The interface builds and the CPU-only production Docker image builds. A 12-move local production test and four-move hosted streaming test returned legal moves with history consistency. HF CPU searches took 9.50–9.53 seconds in the measured hosted run. Chrome confirmed the live interface returns the opening move when playing as Black.

Seven remaining dependency advisories concern the existing Tailwind build tooling and require the frontend migration. Compatible dependency fixes were applied; the full frontend redesign remains deferred.
