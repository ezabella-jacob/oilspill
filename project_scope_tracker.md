# Project Scope Tracker — Marine Oil-Spill Intelligence System

Purpose: keep an honest running record of what's been descoped or deferred from the original plan, so nothing gets quietly lost as the project continues.

## Current status by stage

**Stage 1 — Detection model:** DONE (v1, workable)
- Trained U-Net (ResNet18 encoder, ImageNet pretrained) on Kaggle "Deep-SAR SOS" dataset (Sentinel-1A + ALOS PALSAR combined, 6,455 samples)
- Best val_loss: 0.4344 (epoch 13/15)
- Known weakness: false positives on background-only (clean water) patches

**Stage 2 — Spill characterization:** IN PROGRESS
- Shape/area/orientation extraction from mask: built (`characterize_spill.py`)
- Real-world geolocation: NOT YET WORKING (see descoped item #3 below)

**Stage 3 — Drift/movement modeling:** NOT STARTED

**Stage 4 — AIS correlation:** NOT STARTED

**Stage 5 — Vessel ranking:** NOT STARTED

**Dashboard / deployment:** NOT STARTED

## Descoped or deferred items (revisit before calling this "finished")

1. **Look-alike discrimination class dropped.**
   - Original plan: 3-class detection (oil / look-alike / clean water) using the Zenodo dataset, explicitly called out as the hardest and most interesting part of Stage 1.
   - Current state: binary only (oil / not-oil), because we switched to the Kaggle dataset after Zenodo downloads became unworkable (bandwidth throttling, disk space, session resets).
   - To revisit: once the full pipeline works end-to-end, consider retraining Stage 1 on a look-alike-inclusive dataset (Zenodo, if access improves, or another 3-class source).

2. **3 historical case studies identified but not connected to the pipeline.**
   - Genesis River (Houston Ship Channel, May 2019) — Sentinel-1 coverage confirmed in Copernicus Browser.
   - Orange County spill (Oct 2021) — multi-vessel candidate case (MSC Danit, Beijing), good for ranking evaluation, but has a delayed-source timing quirk.
   - Green Canyon 248 (Gulf of Mexico, May 2016) — open-water, non-vessel source, useful for drift validation only.
   - None of these have been run through the trained model yet. This connection is what makes the project's evaluation section credible — needs to happen before "finished."

3. **No real-world geolocation yet.**
   - Kaggle training patches have no geo metadata (not real georeferenced Sentinel-1 products).
   - Stage 2 currently only computes pixel-space geometry (area in pixels, shape, orientation), not real lat/lon or real-world area.
   - To revisit: real geolocation requires running the model on actual downloaded Sentinel-1 scenes (from Copernicus, as scoped in Phase 0), which do carry real coordinates.

## Guiding principle going forward

Finish the pipeline end-to-end on the current simplified scope first (proves the system works), then circle back to items above before considering the project done. Don't let scope keep quietly eroding stage by stage without it being a conscious decision.