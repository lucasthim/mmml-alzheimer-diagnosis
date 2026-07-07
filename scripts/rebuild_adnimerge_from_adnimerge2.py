#!/usr/bin/env python3
"""Rebuild an ``ADNIMERGE.csv``-equivalent from the ADNIMERGE2 R data package.

Why this exists
---------------
The new ADNI R package (``ADNIMERGE2``, ATRI Biostatistics) no longer ships the
flat "Key ADNI Table Merged Data (adnimerge)" table that this project's pipeline
depends on. The data is now normalized into ~200 per-instrument ``.rda`` tables.
This script reconstructs exactly the columns that
``src/data_preprocessing/cognitive_tests_preprocessing.py`` reads, by joining the
relevant raw tables on ``RID`` + ``VISCODE`` and recomputing the few derived
scores (RAVLT_*, AGE, DX_bl). ``ADASQ4`` is taken from the package's shipped
``inst/extradata/pacc-raw-input/pacc_adas_q4score_long.csv``.

The output is a drop-in for the old ``ADNIMERGE.csv``: feed it to
``execute_cognitive_data_preprocessing`` unchanged.

Usage
-----
    pip install --user rdata pandas
    python scripts/rebuild_adnimerge_from_adnimerge2.py \
        --pkg data/ADNIMERGE2 --out data/tabular/ADNIMERGE.csv --selfcheck

Caveats (see docs/data/data-acquisition.md and docs/data/data-semantics.md)
--------------------------------------------------------------------------
* IMAGEUID is linked INTERNALLY from ADNIMERGE2's own UCSF FreeSurfer / morphometry
  tables (the processed-T1 image id that keys the downloadable "Pre-processed"
  collection) -- see link_imageuid_internal(). No external LONI image-collection
  export is required. The join is (RID, VISCODE) with the screening family
  sc/scmri treated as baseline (bl); 'init' is deliberately NOT collapsed (it is
  the distinct ADNI3 initial visit -- collapsing it was an old bug that glued 2005
  baseline scans onto 2017-2019 visits). Each IMAGEUID lands on exactly one visit,
  so no scan is duplicated across visits. ~65% of diagnosed visits get an IMAGEUID
  (only FreeSurfered visits have one); the rest keep the 999999 "no MRI" sentinel
  downstream. Pass --image-collection to override with your own Subject/Visit/Image
  mapping (e.g. the acquisition-date linker's output) instead.
* Ecog* columns are emitted empty (NaN); the pipeline drops them by default
  (exclude_ecog_tests=True). Populate from ECOGPT/ECOGSP if you need them.
* Joins are on (RID, VISCODE). ADNI's VISCODE/VISCODE2 coding has edge cases;
  a handful of visits may not line up across instruments.
* Category labels are remapped from ADNIMERGE2's full SDTM strings back to the
  classic ADNIMERGE short labels so the unchanged encoders work correctly.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    import rdata
except ImportError:
    sys.exit("Missing dependency: pip install --user rdata")
import numpy as np
import pandas as pd

# ---- classic-ADNIMERGE category vocab (what the encoders in
# cognitive_tests_preprocessing.py compare against) -------------------------
RACE_MAP = {
    "White": "White",
    "Black or African American": "Black",
    "Asian": "Asian",
    "American Indian or Alaskan Native": "Am Indian/Alaskan",
    "Native Hawaiian or Other Pacific Islander": "Hawaiian/Other PI",
    "Other Pacific Islander": "Hawaiian/Other PI",
    "More than one race": "More than one",
    "Unknown": "Unknown",
}
ETH_MAP = {
    "Not Hispanic or Latino": "Not Hisp/Latino",
    "Hispanic or Latino": "Hisp/Latino",
    "Unknown": "Unknown",
}
MARRY_KEEP = {"Married", "Widowed", "Divorced", "Never married"}

ECOG_COLS = [
    "EcogPtMem", "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan",
    "EcogPtDivatt", "EcogPtTotal", "EcogSPMem", "EcogSPLang", "EcogSPVisspat",
    "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "EcogSPTotal",
]

# Final column order = exactly what select_cognitive_data() selects.
OUT_COLS = (
    ["RID", "PTID", "VISCODE", "SITE", "COLPROT", "ORIGPROT", "EXAMDATE",
     "IMAGEUID", "DX", "DX_bl"]
    + ["AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY"]
    + ["CDRSB", "ADAS11", "ADAS13", "ADASQ4", "MMSE", "RAVLT_immediate",
       "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
       "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ", "MOCA"]
    + ECOG_COLS
)


def load_rda(pkg, name):
    """Load one ADNIMERGE2 table to a DataFrame with normalized keys."""
    path = os.path.join(pkg, "data", name + ".rda")
    obj = rdata.read_rda(path)
    df = obj.get(name, next(iter(obj.values())))
    df = pd.DataFrame(df)
    df.columns = [str(c) for c in df.columns]
    if "RID" in df.columns:
        df["RID"] = pd.to_numeric(df["RID"], errors="coerce").astype("Int64")
    if "VISCODE" in df.columns:
        df["VISCODE"] = df["VISCODE"].astype(str).str.strip().str.lower()
    return df


def dedup(df, keep_cols):
    """One row per (RID, VISCODE), preferring rows with the most data."""
    sub = df[["RID", "VISCODE"] + keep_cols].copy()
    sub["_n"] = sub[keep_cols].notna().sum(axis=1)
    sub = sub.sort_values("_n", ascending=False).drop_duplicates(
        ["RID", "VISCODE"], keep="first").drop(columns="_n")
    return sub


def coalesce(*series):
    out = series[0].copy()
    for s in series[1:]:
        out = out.where(out.notna(), s)
    return out


def to_dt(s):
    """Parse a date column. rdata returns R Date columns as numeric days since
    1970-01-01; character dates come through as strings. Handle both."""
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1970-01-01", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


# UCSF FreeSurfer / morphometry tables that carry the PROCESSED-T1 IMAGEUID (the
# I##### that keys the downloadable "Pre-processed" collection) alongside RID +
# VISCODE. One processed scan per visit; listed in preference order (cross-phase /
# most complete first). Only per-visit scan columns are used -- baseline-reference
# columns (FOXLABBSI.LONIUID_BASE, TBM22.IMAGEUID_1) are excluded because they
# repeat one baseline scan across every follow-up row. UCSFFSX6 is excluded: its
# ADNI3 ids are RAW acquisition ids, absent from the processed download namespace.
UCSF_IMAGEUID_TABLES = [
    ("UCSFFSX7", "IMAGEUID"), ("UCSFFSX", "IMAGEUID"), ("UCSFFSX51", "IMAGEUID"),
    ("UCSFFSX51_ADNI1_3T", "IMAGEUID"), ("UCSFFSL", "IMAGEUID"),
    ("UCSFFSL51", "IMAGEUID"), ("UCSFFSL51ALL", "IMAGEUID"), ("UCSFFSL51Y1", "IMAGEUID"),
    ("BSI", "IMAGEUID"), ("UASPMVBM", "IMAGEUID"), ("UCSFSNTVOL", "IMAGEUID"),
    ("FOXLABBSI", "LONIUID"),   # label is LONIUID but values are the IMAGEUID namespace
]


def link_imageuid_internal(spine, pkg):
    """Fill spine['IMAGEUID'] from ADNIMERGE2's own UCSF FreeSurfer / morphometry
    tables -- no external image-collection export needed.

    Join is on (RID, VISCODE) with the screening family sc/scmri normalized to the
    baseline (bl): the screening MRI IS the baseline timepoint. 'init' is NEVER
    collapsed (it is the distinct ADNI3 initial visit; collapsing it was the old
    bug that glued 2005 baseline scans onto 2017-2019 visits). Each IMAGEUID is
    then placed on exactly ONE visit -- the one whose EXAMDATE is nearest the scan's
    own acquisition date (so a screening scan stays on the near sc row rather than a
    far, re-baselined bl row) -- so no scan is duplicated across visits. (A scan
    acquired between two visits during a diagnosis transition can still sit on a
    visit whose label differs from the diagnosis at the scan instant -- ~0.1% of
    links -- but nothing is glued years off as the old init->bl collapse did.)
    Deterministic: nearest scan date, then prefer bl, then earliest exam; map ties
    broken by table priority then lowest IMAGEUID.
    """
    frames = []
    for prio, (name, idcol) in enumerate(UCSF_IMAGEUID_TABLES):
        try:
            t = load_rda(pkg, name)
        except FileNotFoundError:
            continue
        if not {"RID", "VISCODE", idcol} <= set(t.columns):
            continue
        iid = pd.to_numeric(t[idcol].astype(str).str.replace("I", "", regex=False),
                            errors="coerce")
        acq = to_dt(t["EXAMDATE"]) if "EXAMDATE" in t.columns else pd.Series(pd.NaT, index=t.index)
        sub = pd.DataFrame({"RID": t["RID"], "VISCODE": t["VISCODE"],
                            "IMAGEUID": iid, "SCANDATE": acq.values, "_p": prio})
        frames.append(sub[sub["IMAGEUID"] > 0])

    spine = spine.drop(columns=["IMAGEUID"], errors="ignore")
    if not frames:
        print("  WARNING: no UCSF imaging tables found in package; IMAGEUID left empty")
        spine["IMAGEUID"] = np.nan
        return spine

    uni = pd.concat(frames, ignore_index=True)
    uni["IMAGEUID"] = uni["IMAGEUID"].astype("Int64")

    def scbl(s):   # screening family -> baseline; NB: 'init' is NOT collapsed
        return s.replace({"sc": "bl", "scmri": "bl"})
    uni["VK"] = scbl(uni["VISCODE"])
    mp = uni.sort_values(["_p", "IMAGEUID"]).drop_duplicates(["RID", "VK"], keep="first")

    spine["VK"] = scbl(spine["VISCODE"])
    spine = spine.merge(mp[["RID", "VK", "IMAGEUID", "SCANDATE"]], on=["RID", "VK"], how="left")

    # each IMAGEUID on exactly one visit: keep the row whose EXAMDATE is nearest the
    # scan's own acquisition date (so a screening scan stays on the near sc row, not
    # a far re-baselined bl row); fall back to prefer-bl then earliest exam when the
    # scan date is unknown.
    gap = (to_dt(spine["EXAMDATE"]) - spine["SCANDATE"]).abs().dt.days
    vr = spine["VISCODE"].map({"bl": 0, "sc": 1}).fillna(2)
    order = spine.assign(_g=gap.fillna(10 ** 9), _vr=vr,
                         _ex=to_dt(spine["EXAMDATE"]), _i=spine.index) \
                 .sort_values(["_g", "_vr", "_ex"])
    keep = set(order[spine["IMAGEUID"].notna()]
               .drop_duplicates("IMAGEUID", keep="first")["_i"])
    drop = spine["IMAGEUID"].notna() & ~spine.index.isin(keep)
    spine.loc[drop, "IMAGEUID"] = pd.NA
    spine["IMAGEUID"] = pd.to_numeric(spine["IMAGEUID"], errors="coerce")  # float, NaN for missing
    spine = spine.drop(columns=["VK", "SCANDATE"])
    print(f"  IMAGEUID from internal UCSF tables: "
          f"{int(spine['IMAGEUID'].notna().sum())}/{len(spine)} visit rows "
          f"({int(spine['IMAGEUID'].nunique())} distinct scans)")
    return spine


def build(pkg, image_collection=None):
    # ---- spine: DXSUM (per-visit diagnosis + ids) -------------------------
    dx = load_rda(pkg, "DXSUM")
    dx = dx.rename(columns={"DIAGNOSIS": "DX", "SITEID": "SITE"})
    dx = dx[dx["DX"].notna()]
    spine = dedup(dx, ["PTID", "DX", "SITE", "COLPROT", "ORIGPROT", "EXAMDATE"])
    print(f"  spine (DXSUM, non-null DX): {len(spine)} visit rows")

    # ---- demographics (subject grain) -------------------------------------
    pt = load_rda(pkg, "PTDEMOG")
    demo_cols = ["PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY",
                 "PTDOB", "PTDOBYY"]
    demo_cols = [c for c in demo_cols if c in pt.columns]
    demo = (pt.sort_values("VISCODE")
              .groupby("RID")[demo_cols]
              .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
              .reset_index())
    spine = spine.merge(demo, on="RID", how="left")

    # ---- AGE (baseline age, constant per subject) -------------------------
    # DOB is MM/YYYY text; PTDOBYY is the clean 4-digit year. Use the year only
    # (ADNI provides no birth day for privacy); AGE = baseline exam year - birth year.
    exam = to_dt(spine["EXAMDATE"])
    spine["EXAMDATE"] = exam.dt.strftime("%Y-%m-%d")  # normalize for output
    base_year = exam.dt.year.groupby(spine["RID"]).transform("min")
    birth_year = pd.to_numeric(spine.get("PTDOBYY"), errors="coerce")
    if "PTDOB" in spine.columns:
        alt = spine["PTDOB"].astype(str).str.extract(r"(\d{4})")[0]
        birth_year = birth_year.where(birth_year.notna(),
                                      pd.to_numeric(alt, errors="coerce"))
    spine["AGE"] = (base_year - birth_year).astype(float)

    # ---- DX_bl (baseline diagnosis per subject) ---------------------------
    rank = spine["VISCODE"].map({"bl": 0, "sc": 1}).fillna(2)
    bl = (spine.assign(_r=rank, _e=exam)
                .sort_values(["_r", "_e"])
                .drop_duplicates("RID", keep="first")[["RID", "DX"]]
                .rename(columns={"DX": "DX_bl"}))
    bl["DX_bl"] = bl["DX_bl"].replace({"Dementia": "AD"})
    spine = spine.merge(bl, on="RID", how="left")

    # ---- cognitive instrument scores (visit grain) ------------------------
    def merge_scores(spine, name, mapping, src_cols=None, derive=None):
        # src_cols = raw columns to retain before deriving; defaults to the
        # mapping keys (correct for direct 1:1 renames).
        t = load_rda(pkg, name)
        src_cols = list(mapping) if src_cols is None else src_cols
        need = [c for c in src_cols if c in t.columns]
        t2 = dedup(t, need) if need else t[["RID", "VISCODE"]]
        if derive:
            t2 = derive(t2)
        t2 = t2.rename(columns={k: v for k, v in mapping.items() if k in t2.columns})
        keep = ["RID", "VISCODE"] + [v for v in mapping.values() if v in t2.columns]
        return spine.merge(t2[keep], on="RID VISCODE".split(), how="left")

    spine = merge_scores(spine, "CDR", {"CDRSB": "CDRSB"})
    spine = merge_scores(spine, "ADAS", {"TOTSCORE": "ADAS11", "TOTAL13": "ADAS13"})
    spine = merge_scores(spine, "MMSE", {"MMSCORE": "MMSE"})
    spine = merge_scores(spine, "MOCA", {"MOCA": "MOCA"})
    spine = merge_scores(spine, "FAQ", {"FAQTOTAL": "FAQ"})

    neurobat_src = ["AVTOT1", "AVTOT2", "AVTOT3", "AVTOT4", "AVTOT5", "AVDEL30MIN",
                    "LDELTOTAL", "DIGITSCOR", "TRABSCOR"]

    def ravlt(df):
        # ensure every source column exists (NaN if the table lacked it)
        for c in neurobat_src:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        t5, t1, d30 = df["AVTOT5"], df["AVTOT1"], df["AVDEL30MIN"]
        df["RAVLT_immediate"] = df[["AVTOT1", "AVTOT2", "AVTOT3", "AVTOT4",
                                    "AVTOT5"]].sum(axis=1, min_count=1)
        df["RAVLT_learning"] = t5 - t1
        df["RAVLT_forgetting"] = t5 - d30
        df["RAVLT_perc_forgetting"] = (
            100.0 * (t5 - d30) / t5).replace([np.inf, -np.inf], np.nan)
        return df

    spine = merge_scores(
        spine, "NEUROBAT",
        {"RAVLT_immediate": "RAVLT_immediate", "RAVLT_learning": "RAVLT_learning",
         "RAVLT_forgetting": "RAVLT_forgetting",
         "RAVLT_perc_forgetting": "RAVLT_perc_forgetting",
         "LDELTOTAL": "LDELTOTAL", "DIGITSCOR": "DIGITSCOR", "TRABSCOR": "TRABSCOR"},
        src_cols=neurobat_src, derive=ravlt)

    # ---- ADASQ4 from the shipped PACC long CSV ----------------------------
    q4_path = os.path.join(pkg, "inst", "extradata", "pacc-raw-input",
                           "pacc_adas_q4score_long.csv")
    if os.path.exists(q4_path):
        q4 = pd.read_csv(q4_path)
        q4["RID"] = pd.to_numeric(q4["RID"], errors="coerce").astype("Int64")
        q4["VISCODE"] = q4["VISCODE"].astype(str).str.strip().str.lower()
        q4 = q4.dropna(subset=["SCORE"]).drop_duplicates(["RID", "VISCODE"])
        spine = spine.merge(q4[["RID", "VISCODE", "SCORE"]].rename(
            columns={"SCORE": "ADASQ4"}), on=["RID", "VISCODE"], how="left")
    else:
        print(f"  WARNING: ADASQ4 source not found at {q4_path}; emitting NaN")
        spine["ADASQ4"] = np.nan

    # ---- category remap to classic vocab ----------------------------------
    def map_race(v):
        if not isinstance(v, str):
            return v
        if "|" in v:
            return "More than one"
        if v in RACE_MAP:
            return RACE_MAP[v]
        if "Hawaiian" in v or "Pacific" in v:
            return "Hawaiian/Other PI"
        if "Indian" in v or "Alask" in v:
            return "Am Indian/Alaskan"
        return v

    if "PTRACCAT" in spine:
        spine["PTRACCAT"] = spine["PTRACCAT"].apply(map_race)
    if "PTETHCAT" in spine:
        spine["PTETHCAT"] = spine["PTETHCAT"].map(ETH_MAP).fillna(spine["PTETHCAT"])
    if "PTMARRY" in spine:
        spine["PTMARRY"] = spine["PTMARRY"].apply(
            lambda v: v if v in MARRY_KEEP else "Unknown")

    # ---- IMAGEUID --------------------------------------------------------
    # Default: link internally from ADNIMERGE2's own UCSF FreeSurfer tables (the
    # processed-T1 image id that keys the downloadable collection). Optional
    # override: --image-collection, a Subject/Visit/Image Data ID mapping (e.g. a
    # LONI export or the acquisition-date linker's output), matched on PTID+VISCODE.
    if image_collection and os.path.exists(image_collection):
        ic = pd.read_csv(image_collection)
        cols = {c.lower(): c for c in ic.columns}
        subj = cols.get("subject") or cols.get("ptid")
        img = cols.get("image data id") or cols.get("imageuid") or cols.get("image_data_id")
        vis = cols.get("visit") or cols.get("viscode")
        spine = spine.drop(columns=["IMAGEUID"], errors="ignore")
        if subj and img:
            ic = ic.rename(columns={subj: "PTID", img: "IMAGEUID"})
            ic["IMAGEUID"] = pd.to_numeric(
                ic["IMAGEUID"].astype(str).str.replace("I", "", regex=False), errors="coerce")
            key = ["PTID"] + (["VISCODE"] if vis else [])
            if vis:
                ic = ic.rename(columns={vis: "VISCODE"})
                ic["VISCODE"] = ic["VISCODE"].astype(str).str.strip().str.lower()
            ic = ic.dropna(subset=["IMAGEUID"]).drop_duplicates(key)
            spine = spine.merge(ic[key + ["IMAGEUID"]], on=key, how="left")
            print(f"  IMAGEUID from --image-collection: "
                  f"{int(spine['IMAGEUID'].notna().sum())}/{len(spine)} rows")
        else:
            print("  WARNING: image-collection CSV lacks Subject/Image columns; "
                  "IMAGEUID left empty")
            spine["IMAGEUID"] = np.nan
    else:
        spine = link_imageuid_internal(spine, pkg)

    # ---- finalize ---------------------------------------------------------
    for c in ECOG_COLS:
        spine[c] = np.nan
    for c in OUT_COLS:
        if c not in spine.columns:
            spine[c] = np.nan
    return spine[OUT_COLS]


def selfcheck(df):
    """Mimic cognitive_tests_preprocessing.py to prove the CSV is a drop-in."""
    print("\n=== SELF-CHECK (simulating cognitive_tests_preprocessing) ===")
    id_cols = ["RID", "PTID", "VISCODE", "SITE", "COLPROT", "ORIGPROT",
               "EXAMDATE", "IMAGEUID", "DX", "DX_bl"]
    demo = ["AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY"]
    neuro = ["CDRSB", "ADAS11", "ADAS13", "ADASQ4", "MMSE", "RAVLT_immediate",
             "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
             "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ", "MOCA"] + ECOG_COLS
    missing = [c for c in id_cols + demo + neuro if c not in df.columns]
    assert not missing, f"MISSING COLUMNS (pipeline would KeyError): {missing}"
    print("  column contract: OK (all id/demographic/neuropsych columns present)")
    d = df[~df["DX"].isnull()].copy()
    d["DX"] = d["DX"].replace({"Dementia": "AD"})
    dist = d["DX"].value_counts(dropna=False).to_dict()
    print(f"  rows with diagnosis: {len(d)}")
    print(f"  DX distribution (->CN/MCI/AD): {dist}")
    enc_ok = set(d["DX"].dropna().unique()) <= {"CN", "MCI", "AD"}
    print(f"  DX values encodable to 0/1/2: {enc_ok}")
    for col, label in [("PTGENDER", "Male/Female"), ("PTRACCAT", "White/Black/Asian/..."),
                       ("PTETHCAT", "Hisp labels"), ("PTMARRY", "marital labels")]:
        vals = [str(v) for v in d[col].dropna().unique()][:8]
        print(f"  {col} ({label}): {vals}")
    print("  non-null coverage of key scores:")
    for c in ["ADASQ4", "RAVLT_immediate", "CDRSB", "MMSE", "MOCA", "FAQ", "AGE"]:
        print(f"     {c:22s}: {int(d[c].notna().sum())}/{len(d)}")
    print(f"  IMAGEUID present (else ->999999 sentinel): {int(d['IMAGEUID'].notna().sum())}/{len(d)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", default="data/ADNIMERGE2",
                    help="path to the unpacked ADNIMERGE2 package")
    ap.add_argument("--out", default="data/tabular/ADNIMERGE.csv",
                    help="output ADNIMERGE.csv path")
    ap.add_argument("--image-collection", default=None,
                    help="optional override: a Subject/Visit/Image Data ID CSV (LONI "
                         "export or the acqdate linker's output) to source IMAGEUID from "
                         "instead of the internal UCSF FreeSurfer tables")
    ap.add_argument("--selfcheck", action="store_true",
                    help="simulate the downstream preprocessing to validate")
    args = ap.parse_args()

    print(f"Building ADNIMERGE.csv-equivalent from {args.pkg} ...")
    df = build(args.pkg, image_collection=args.image_collection)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {args.out}")
    if args.selfcheck:
        selfcheck(df)


if __name__ == "__main__":
    main()
