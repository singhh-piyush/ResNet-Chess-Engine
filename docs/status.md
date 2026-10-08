# Backend repair status

- Legacy source, weights, datasets, logs and generated bundles preserved under ignored `archive/`.
- New promotion-safe shared encoding, validated API, bounded neural search and real progress streaming implemented.
- Existing fold-0 model is a baseline, not a newly verified improved model.
- Refreshed public archive: 3,457 games. Offline relabeling is in progress.
- GPU retraining and independent quality evaluation remain required before candidate promotion.
- Hosting: existing free CPU HF Space; GitHub main deploys through the tested workflow with a scoped trusted publisher.

Validation: 11 backend/training checks passed; interface builds. Local baseline CPU search: 9.52 seconds at the initial position. Seven remaining dependency advisories concern the existing Tailwind build tooling and require the frontend migration.
