Put the cloned/downloaded IO-VNBD tree here, for example:

data/raw/IO-VNBD/Synchronised V abd S datasets/...

The GitHub repo uses Git LFS for CSV bodies. A 134-byte file is a pointer, not
the recording. After clone, run `git lfs pull` (or download the zip via LFS).

Phase 0 runs on data/fixtures/ until this folder is populated.
