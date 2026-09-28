"""Load a differential abundance analysis (DAA) table and map it onto model
metabolites (protocol §1.2-1.3)."""
import re

import numpy as np
import pandas as pd


def _norm_hmdb(x):
    m = re.fullmatch(r"HMDB0*(\d+)", x, re.I)
    return f"HMDB{int(m.group(1)):07d}" if m else None


def _norm_chebi(x):
    m = re.fullmatch(r"(?:CHEBI:)?(\d+)", x, re.I)
    return f"CHEBI:{m.group(1)}" if m else None


def build_lookup(model):
    """Map every recognisable identifier to a compartment-free met_id."""
    lookup = {}
    for s in model.species.values():
        keys = [s.met_id, s.id, s.name.lower()]
        keys += [_norm_hmdb(h) for h in s.xrefs.get("hmdb", [])]
        keys += s.xrefs.get("kegg", [])
        keys += [_norm_chebi(c) for c in s.xrefs.get("chebi", [])]
        for k in keys:
            if k:
                lookup.setdefault(k, s.met_id)
    return lookup


def map_identifier(raw, lookup):
    x = str(raw).strip()
    candidates = [x, x.lower(), _norm_hmdb(x), _norm_chebi(x) if x.upper().startswith("CHEBI") else None]
    # Human-GEM ids with a compartment suffix or the SBML prefix: MAM01570c, M_MAM01570c
    m = re.fullmatch(r"(?:M_)?(MAM\d{5})[a-z]?", x)
    if m:
        candidates.insert(0, m.group(1))
    for c in candidates:
        if c and c in lookup:
            return lookup[c]
    return None


def load_daa(path, model, id_cols, fc_col, p_col, alpha, fc_is_linear=False, min_abs_log2fc=0.0):
    """id_cols: identifier columns tried in order (e.g. HMDB, then KEGG, then
    name); the first that maps to a model metabolite wins."""
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    for col in (*id_cols, fc_col, p_col):
        if col not in df.columns:
            raise SystemExit(f"DAA table {path} has no column '{col}' (columns: {list(df.columns)})")

    lookup = build_lookup(model)
    ids = df[list(id_cols)]
    df = df[[fc_col, p_col]].rename(columns={fc_col: "log2fc", p_col: "p"})
    df.insert(0, "input_id", ids.iloc[:, 0].astype(str))
    df["met_id"], df["matched_on"] = None, ""
    for col in id_cols:
        todo = df["met_id"].isna() & ids[col].notna()
        hits = ids.loc[todo, col].map(lambda x: map_identifier(x, lookup))
        found = hits.dropna().index
        df.loc[found, "met_id"] = hits[found]
        df.loc[found, "matched_on"] = col
        df.loc[found, "input_id"] = ids.loc[found, col].astype(str)
    if fc_is_linear:
        df["log2fc"] = np.log2(df["log2fc"])
    df = df.dropna(subset=["log2fc", "p"])
    unmapped = df[df["met_id"].isna()]
    df = df.dropna(subset=["met_id"])

    # Several input rows can hit the same model metabolite (e.g. a name and
    # an HMDB id); keep the most significant one.
    duplicates = df[df.duplicated("met_id", keep=False)]
    df = df.sort_values("p").drop_duplicates("met_id", keep="first")

    df["phi"] = df["log2fc"].abs()                     # §1.2 initial voltage
    df["significant"] = (df["p"] < alpha) & (df["log2fc"].abs() >= min_abs_log2fc)
    df["role"] = np.where(~df["significant"], "",
                          np.where(df["log2fc"] > 0, "source", np.where(df["log2fc"] < 0, "target", "")))
    return df.reset_index(drop=True), unmapped, duplicates
