# data

Fetch scripts only. Downloaded files go to `data/raw/`, which git ignores.
Each download gets a `.sha256` file beside it, so a result can name the
exact bytes it used.

The project-wide data layer is `qp_data` in `python/data` (retries, a
content-addressed store in `data/store/`, a manifest, publication lags and
as-of queries; see `python/data/README.md`). `data/fetch/qp_fetch.py` stays
for the research scripts that import it by path.

| Source | Access | Status |
|---|---|---|
| ECB Data Portal, EXR reference rates | open API, no key; reuse with attribution to the ECB | used: `fetch_ecb_fx` |
| FRED `fredgraph.csv` | open, no key; Fed and BLS series are public domain | available: `fetch_fred`, allow-list of public-domain series; anything else (third-party series such as `SP500`) is refused |
| Nasdaq Data Link (incl. Sharadar) | API key needed, so an account and its terms | **excluded by the owner** (no accounts, no payment details). The fetcher stub was removed in v2.1 |
| LOBSTER sample files | download page behind LOBSTER's terms of use | **excluded by the owner**. The parser was removed in v2.1 |

Fetch the training-period ECB file:

```bash
python data/fetch/qp_fetch.py ecb
```

The default end date is 2021-12-31. Fetching the 2022-2025 hold-out is not
blocked here, but no study may read it before its pre-registration is
signed (see `research/README.md`).
