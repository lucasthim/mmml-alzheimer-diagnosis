*Part of the [MMML-Alzheimer documentation](../README.md). How to obtain the tabular data now that ADNI ships it as the ADNIMERGE2 R package instead of a flat `ADNIMERGE.csv`, and how to rebuild the file this project expects.*

# ADNIMERGE2 → `ADNIMERGE.csv` (rebuilding the tabular data)

**The single most important change since this project was written:** ADNI no longer
distributes the flat *Key ADNI Table Merged Data* (`adnimerge`) table that this
codebase depends on. It now ships the whole study as the **ADNIMERGE2 R data
package** (ATRI Biostatistics), a normalized set of ~200 per-instrument `.rda`
tables. From the package methods PDF:

> *"the 'Key ADNI Table Merged Data (adnimerge)' will no longer be provided in the
> new version of the ADNI R package."*

So the pipeline's input — `data/tabular/ADNIMERGE.csv`, read by
[cognitive_tests_preprocessing.py#L23](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L23)
— must be **reconstructed** by joining the per-instrument tables. This project
ships a script that does exactly that:
[scripts/rebuild_adnimerge_from_adnimerge2.py](../../scripts/rebuild_adnimerge_from_adnimerge2.py).

> **TL;DR**
> ```bash
> pip install --user rdata pandas
> # 1. (one-time) build the IMAGEUID map from the UCSF FreeSurfer tables — see §5
> # 2. rebuild the flat file:
> python3 scripts/rebuild_adnimerge_from_adnimerge2.py \
>     --pkg data/ADNIMERGE2 --out data/tabular/ADNIMERGE.csv \
>     --imageuid-map data/reference/IMAGEUID_FROM_UCSF.csv --selfcheck
> ```
> Output: `data/tabular/ADNIMERGE.csv` (15,836 visit-rows × 44 cols), a drop-in
> for the existing [cognitive_tests_preprocessing.py](../../src/data_preprocessing/cognitive_tests_preprocessing.py).

## 1. What ADNIMERGE2 is

- An R **data package** (`ADNIMERGE2`, v0.1.1, ATRI Biostatistics), obtained from
  LONI like any other ADNI data (same DUA — see [data-acquisition.md](data-acquisition.md)).
  It unpacks to `data/ADNIMERGE2/` (gitignored).
- **~200 raw tables** as `.rda` files under `data/ADNIMERGE2/data/`, one per ADNI
  instrument, uppercase-named: `ADAS`, `MMSE`, `MOCA`, `CDR`, `FAQ`, `NEUROBAT`,
  `PTDEMOG`, `DXSUM`, `REGISTRY`, `APOERES`, the UCSF FreeSurfer imaging tables, etc.
- A **36,898-row data dictionary** (`DATADIC.rda`: `TBLNAME`, `FLDNAME`, `TEXT`, `CODE`).
- **Derived CDISC/ADaM "analysis datasets"** (`ADSL` subject-level, `ADQS`
  questionnaire-analysis) built with the pharmaverse framework, plus a modified
  PACC score dataset.
- **Covers ADNI1, GO, 2, 3, and 4** — the previous package stopped at ADNI3, so
  ADNI4 is new (see [§6](#6-phase-coverage-adni1-to-adni4)).
- Values are **pre-labeled**, matching what the old code expects as strings:
  `DIAGNOSIS` ∈ {`CN`,`MCI`,`Dementia`}, `PTGENDER` ∈ {`Male`,`Female`}, etc.

## 2. Reading the `.rda` tables

Use the **pure-Python `rdata`** package (no R, no compiler):

```python
import rdata
df = rdata.read_rda("data/ADNIMERGE2/data/ADAS.rda")["ADAS"]   # -> pandas DataFrame
```

Two gotchas worth knowing (both handled by the rebuild script, both catalogued in
[known-issues.md](../reference/known-issues.md)):

- **R `Date` columns arrive as numeric days since 1970-01-01.** `pd.to_datetime`
  will misread them as nanoseconds (every date → 1970). Parse with
  `pd.to_datetime(col, unit="D", origin="1970-01-01")`.
- **The derived `ADSL` / `ADQS` ADaM datasets do not parse in pure Python**
  (`rdata` raises `249 is not a valid RObjectType` — they use R `labelled`
  attributes). They need actual R (`load("ADSL.rda")`). `pyreadr` is the other
  Python option but it failed to build in this environment. **All raw instrument
  tables read fine in `rdata`** — the rebuild only needs raw tables.

## 3. The rebuild script

[scripts/rebuild_adnimerge_from_adnimerge2.py](../../scripts/rebuild_adnimerge_from_adnimerge2.py)
joins the raw tables on `RID` + `VISCODE`, recomputes the derived scores, remaps
category labels, and emits a flat CSV matching the exact column contract of
[select_cognitive_data](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L69).
`--selfcheck` simulates the downstream preprocessing to prove the output is a
drop-in (column contract, DX encodable to 0/1/2, score coverage).

Spine = `DXSUM` (one row per visit with a non-null `DIAGNOSIS`). Demographics are
broadcast per subject from `PTDEMOG`; scores are left-joined per visit.

## 4. Column map: where each pipeline column now comes from

The columns [select_cognitive_data](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L69)
needs, and their ADNIMERGE2 source. **Direct** = column exists; **Derived** =
recomputed (the way classic ADNIMERGE did). Full clinical meanings stay in
[data-semantics.md](data-semantics.md).

| Pipeline column | ADNIMERGE2 source | Direct/Derived | Note |
|---|---|---|---|
| `PTID`,`RID`,`VISCODE`,`COLPROT`,`ORIGPROT` | every table | Direct | join keys / phase |
| `SITE` | `*.SITEID` | Direct | renamed |
| `EXAMDATE` | `DXSUM` (R Date → parsed) | Direct | |
| `DX` | `DXSUM.DIAGNOSIS` | Direct | `Dementia`→`AD` downstream |
| `DX_bl` | `DXSUM` | **Derived** | baseline-visit `DIAGNOSIS` per subject |
| `PTGENDER`,`PTEDUCAT`,`PTETHCAT`,`PTRACCAT`,`PTMARRY` | `PTDEMOG` | Direct | labels **remapped** to classic vocab* |
| `AGE` | `PTDEMOG.PTDOBYY` + baseline `EXAMDATE` | **Derived** | baseline exam year − birth year |
| `CDRSB` | `CDR.CDRSB` | Direct | |
| `ADAS11` / `ADAS13` | `ADAS.TOTSCORE` / `ADAS.TOTAL13` | Direct | renamed |
| `ADASQ4` | `inst/extradata/pacc-raw-input/pacc_adas_q4score_long.csv` | Direct | shipped PACC CSV — see [§4.1](#41-adasq4) |
| `MMSE` | `MMSE.MMSCORE` | Direct | |
| `MOCA` | `MOCA.MOCA` | Direct | only ADNIGO/2 onward |
| `RAVLT_immediate` | `NEUROBAT.AVTOT1..AVTOT5` | **Derived** | sum of 5 trials |
| `RAVLT_learning` | `NEUROBAT` | **Derived** | `AVTOT5 − AVTOT1` |
| `RAVLT_forgetting` | `NEUROBAT` | **Derived** | `AVTOT5 − AVDEL30MIN` |
| `RAVLT_perc_forgetting` | `NEUROBAT` | **Derived** | `100·(AVTOT5−AVDEL30MIN)/AVTOT5` |
| `LDELTOTAL`,`DIGITSCOR`,`TRABSCOR` | `NEUROBAT` | Direct | first two dropped by default |
| `FAQ` | `FAQ.FAQTOTAL` | Direct | renamed |
| `Ecog*` (14 cols) | `ECOGPT`/`ECOGSP` | emitted empty (NaN) | dropped by default (`exclude_ecog_tests=True`) |
| `IMAGEUID` | **not in clinical tables** | merged separately | see [§5](#5-imageuid-the-mri-link) |
| `APOE4` | `APOERES.GENOTYPE` | derivable | *not used by the pipeline* |

\* The encoders in [cognitive_tests_preprocessing.py#L84](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L84)
compare against classic short labels (`'Black'`, `'Not Hisp/Latino'`). ADNIMERGE2
uses full SDTM labels (`'Black or African American'`, `'Not Hispanic or Latino'`),
so the script remaps them — otherwise the race/ethnicity one-hots silently
produce all-zeros.

### 4.1 ADASQ4

The summary `ADAS` table holds only `TOTSCORE`/`TOTAL13` (not Q4). ADASQ4 is
recovered from the package's shipped **`inst/extradata/pacc-raw-input/pacc_adas_q4score_long.csv`**
(`RID, VISCODE, SCORE`; `SCORE_SOURCE=ADASQ4SCORE`) — 12,640 of 15,836 rows. (The
SDTM `QS` table only has ADAS totals; `ITEM` has per-word raw responses.)

## 5. IMAGEUID (the MRI link)

`IMAGEUID` (the `I######` that ties a visit to its MRI and appears in the NIfTI
filenames) is **not** in ADNIMERGE2's clinical/visit tables. The MRI CRF tables
expose `LONIUID` / `LONI_IMAGE`, which is a **different identifier namespace**
(zero overlap with IMAGEUID — do not use it). The real `IMAGEUID` lives in the
**UCSF FreeSurfer** tables (`UCSFFSX`, `UCSFFSX51`, `UCSFFSX51_ADNI1_3T`,
`UCSFFSX6`, `UCSFFSX7`) — i.e. the analysis-grade T1 MP-RAGE that was FreeSurfered,
one per visit.

Build a reusable `RID,VISCODE,IMAGEUID` map by unioning those tables (the snippet
that produced `data/reference/IMAGEUID_FROM_UCSF.csv` — 12,484 pairs — is in the
project history), then merge it during the rebuild:

```bash
python3 scripts/rebuild_adnimerge_from_adnimerge2.py --pkg data/ADNIMERGE2 \
    --out data/tabular/ADNIMERGE.csv --imageuid-map data/reference/IMAGEUID_FROM_UCSF.csv
```

The merge is on `RID`+`VISCODE`, collapsing the screening/baseline family
(`sc`/`scmri`/`bl`/`init`) to one baseline timepoint so a screening scan links to
the baseline diagnosis. Result: **10,862/15,836 (69%)** visit-rows get an IMAGEUID;
the rest stay as the `999999` "no MRI" sentinel ([encode_variables](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L97)).
Verified: 100% of populated IDs trace back to the source map, 10,066 distinct
scans, and `mri_selection` would emit **5,799 CN/AD scans** to download (vs 0
before the merge).

**Caveat — which IMAGEUID this is.** These are the FreeSurfer-processed T1 per
visit: a valid "which scan to download/use" list, **not guaranteed to be the exact
scans downloaded for the original 2021 study**. For a fresh run that is fine (any
valid T1 MP-RAGE per visit works). If you have a real **LONI image-collection
export** instead (the `MPRAGE_REFERENCE` flow in [data-acquisition.md](data-acquisition.md)),
pass it with `--image-collection <csv>` to match on `Subject`+`Visit` and strip
the `I` prefix.

## 6. Phase coverage (ADNI1 to ADNI4)

The rebuilt file spans **all five phases**, including the new **ADNI4**. Distinct
scans (IMAGEUIDs) by collection phase (`COLPROT`):

| Phase | distinct scans | IMAGEUID range |
|---|---|---|
| ADNI1 | 3,568 | 31,863 – 1,189,749 |
| ADNIGO | 414 | 176,283 – 1,221,690 |
| ADNI2 | 2,845 | 225,569 – 1,051,225 |
| ADNI3 | 2,321 | 32,762 – 10,298,187 |
| ADNI4 | 1,049 | 10,251,970 – 11,452,777 |

**ADNI4 did not exist when the original study was built (ADNI1/GO/2/3)**, so this
cohort is broader than the original. To reproduce the original cohort, drop ADNI4
(`df = df[df["COLPROT"] != "ADNI4"]`, or filter `ORIGPROT`); to expand the dataset
for new work, keep it. Per-phase IMAGEUID coverage varies (ADNI1 ~92%, ADNI3 ~84%,
ADNI2/ADNI4 ~50% — visit-code mismatches and visits with no FreeSurfer T1).

## 7. Files this produces

| File | What |
|---|---|
| `data/ADNIMERGE2/` | the unpacked R data package (gitignored) |
| `data/reference/IMAGEUID_FROM_UCSF.csv` | reusable `RID,VISCODE,IMAGEUID` map (12,484 pairs) |
| `data/tabular/ADNIMERGE.csv` | the rebuilt flat file (15,836 × 44) — pipeline input |

After this, continue with the normal flow: run
[cognitive_tests_preprocessing.py](../../src/data_preprocessing/cognitive_tests_preprocessing.py)
→ `COGNITIVE_DATA_PREPROCESSED.csv`, then MRI selection — see
[running-experiments.md](../experiments/running-experiments.md).

## See also

- [data-acquisition.md](data-acquisition.md) — the broader re-download guide (MRI images, atlas)
- [data-semantics.md](data-semantics.md) — full clinical meaning of every column
- [data-structure.md](data-structure.md) — on-disk layout and file catalogue
- [running-experiments.md](../experiments/running-experiments.md) — the end-to-end runbook
- [known-issues.md](../reference/known-issues.md) — the `rdata` Date gotcha, ADSL/ADQS, and more
- [data-overview.md](data-overview.md) — the data landscape hub
