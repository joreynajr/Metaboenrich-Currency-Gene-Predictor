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


def _norm_name(x):
    x = str(x).strip().lower().replace("’", "'")
    x = re.sub(r"[*†‡#]+$", "", x).strip()        # footnote markers
    return re.sub(r"\s+", " ", x)


def name_variants(x):
    """The name, then without a trailing parenthetical: 'phenyllactate (pla)' -> 'phenyllactate'."""
    n = _norm_name(x)
    out = [n]
    stripped = re.sub(r"\s+\([^()]*\)$", "", n)
    if stripped and stripped != n:
        out.append(stripped)
    return out


# cis/trans are not stripped: they name different compounds.
_PREFIX = re.compile(r"^(?:\([rs0-9,]+\)|[ldo]|dl)-", re.I)
LOOSE_MIN_LENGTH = 5          # 'pi', 'gsh' and the like are too short to match loosely
_ACID_FORMS = [("ic acid", "ate"), ("butanoate", "butyrate"), ("propanoate", "propionate"),
               ("pentanoate", "valerate"), ("ethanoate", "acetate")]


def loose_key(name):
    """Name with stereo/position prefixes, punctuation and spacing removed and
    acid/salt forms unified: 'O-acetylcarnitine' and 'Acetylcarnitine' ->
    'acetylcarnitine'; '(R)-3-hydroxybutanoate' and '3-Hydroxybutyric acid'
    -> '3hydroxybutyrate'."""
    n = _norm_name(name)
    n = re.sub(r"\s+\([^()]*\)$", "", n)             # trailing ' (abbreviation)', not 'PI(20:0)'
    prev = None
    while prev != n:
        prev, n = n, _PREFIX.sub("", n)
    for a, b in _ACID_FORMS:
        n = n.replace(a, b)
    return re.sub(r"[^a-z0-9]", "", n)


def build_loose_index(model):
    """loose_key -> met_id, for keys that identify exactly one metabolite."""
    seen = {}
    for s in model.species.values():
        seen.setdefault(loose_key(s.name), set()).add(s.met_id)
    return {k: next(iter(v)) for k, v in seen.items() if len(k) >= LOOSE_MIN_LENGTH and len(v) == 1}


def load_name_bridge(path):
    """Synonym -> [KEGG / HMDB ids] from an id_translation table with columns
    KEGG, HMDB and NAME1..NAMEn (the lab's KEGG/HMDB/PubChem synonym table)."""
    tr = pd.read_csv(path, dtype=str, encoding="utf-8", encoding_errors="replace")
    name_cols = [c for c in tr.columns if c.upper().startswith("NAME")]
    bridge = {}
    for row in tr[["KEGG", "HMDB"] + name_cols].itertuples(index=False):
        ids = [x for x in (row[0], row[1]) if isinstance(x, str) and x.strip()]
        if not ids:
            continue
        for name in row[2:]:
            if isinstance(name, str) and name.strip():
                lst = bridge.setdefault(_norm_name(name), [])
                for i in ids:
                    if i not in lst:
                        lst.append(i)
    return bridge


def load_daa(path, model, id_cols, fc_col, p_col, alpha, fc_is_linear=False, min_abs_log2fc=0.0,
             name_bridge=None):
    """id_cols: identifier columns tried in order (e.g. HMDB, then KEGG, then
    name); the first that maps to a model metabolite wins. Rows still unmapped
    are then looked up by name in `name_bridge` (see load_name_bridge), whose
    KEGG/HMDB ids are mapped in turn. `path` may also be a DataFrame."""
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
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
    if name_bridge:
        for col in id_cols:
            todo = df.index[df["met_id"].isna() & ids[col].notna()]
            for i in todo:
                for variant in name_variants(ids.at[i, col]):
                    hit = next((m for m in (map_identifier(x, lookup) for x in name_bridge.get(variant, [])) if m), None)
                    if hit:
                        df.at[i, "met_id"], df.at[i, "matched_on"] = hit, f"{col} via id_translation"
                        df.at[i, "input_id"] = str(ids.at[i, col])
                        break
    # Last resort: a loose name match, accepted only when it is unambiguous.
    loose = build_loose_index(model)
    for col in id_cols:
        todo = df.index[df["met_id"].isna() & ids[col].notna()]
        for i in todo:
            hit = loose.get(loose_key(ids.at[i, col]))
            if hit:
                df.at[i, "met_id"], df.at[i, "matched_on"] = hit, f"{col} (loose name)"
                df.at[i, "input_id"] = str(ids.at[i, col])
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
