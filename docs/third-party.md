# Third-party source and model dependencies

Kronos upstream: https://github.com/shiyu-coder/Kronos

Pinned source commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.

`wallex_lab/vendor_kronos/` contains the adapted tokenizer/model definitions and its upstream license. `ORIGINS.json` records original and copied hashes and the described changes. `third_party/kronos-67b630e/model/kronos.py` retains the pinned original source required by the genuine sampling adapter. Its SHA-256 is `0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032`. The upstream license is retained alongside it.

Kronos pretrained model/tokenizer weights and Chronos-2 upstream model weights have their own upstream model terms and are not bundled. The small bundled `models/barrier-survival/late` ensemble is the project's trained survival model; it is not a renamed foundation-model checkpoint.

Python dependencies remain governed by their individual licenses. This document does not grant a new blanket license for the repository or override upstream terms.
