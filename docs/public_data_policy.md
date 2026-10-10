# Public data policy

This public portfolio contains generic engineering code and documentation for public-data experiments. Private research topics, institutional data, private reports, identifying metadata and credentials are excluded.

Download source images independently and keep them under `data/raw/`. Keep local outputs under `outputs/`; both are excluded from Git. Environments, model weights, Slicer scenes, Windows diagnostic exports and generated images are also ignored. No data-derived images are published in the initial commit because redistribution terms have not been verified.

Before publishing a later addition, inspect the staged file list and complete staged contents. Confirm source terms, metadata, annotations and whether redistribution of each derived figure is permitted. A `.gitignore` rule does not remove previously tracked content or rewrite history.

The AVT source link is provided in the main README. The local archive version and applicable reuse/redistribution terms remain unverified. Credit datasets, annotations and pretrained software separately. Do not describe model inference as training or technical checks as clinical validation.
