# data

Fetch scripts only. Downloaded files go to `data/raw/`, which git ignores.
Each download gets a `.sha256` file beside it, so a result can name the
exact bytes it used.

| Source | Access | Status |
|---|---|---|
| ECB Data Portal, EXR reference rates | open API, no key; reuse with attribution to the ECB | used: `fetch_ecb_fx` |
| FRED `fredgraph.csv` | open, no key; Fed H.10 (`DEX*`) series are public domain | available: `fetch_fred`; third-party series such as `SP500` are refused |
| Nasdaq Data Link (incl. Sharadar) | API key needed, so an account and its terms; returns 403 without one | **blocked**, needs the maintainer to accept those terms |
| LOBSTER sample files | download page behind LOBSTER's terms of use | **blocked**; the owner accepts the terms and copies the files in by hand, then `read_lobster_sample` parses them |

Fetch the training-period ECB file:

```bash
python data/fetch/qp_fetch.py ecb
```

The default end date is 2021-12-31. Fetching the 2022-2025 hold-out is not
blocked here, but no study may read it before its pre-registration is
signed (see `research/README.md`).
