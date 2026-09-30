"""Build the Pruning Atlas page (one self-contained HTML file).

Needs, from the repository root:
    python -m metaboenrich.structure --out results/structure
    python -m metaboenrich.stage_diagrams --out results/stage_diagrams

Then:
    python tools/atlas/build_atlas.py --id-translation PATH/TO/id_translation.csv \
        --out results/pruning_atlas.html

The id_translation table (KEGG, HMDB, NAME1..NAMEn) adds KEGG/HMDB ids and
synonyms that Human-GEM lacks, so the page's search accepts them; only the
derived lookup is embedded, not the table itself.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from metaboenrich.gem import load_sbml            # noqa: E402
from metaboenrich.network import build_graph      # noqa: E402
from metaboenrich.structure import stages         # noqa: E402


def hmdb(x):
    m = re.fullmatch(r"HMDB0*(\d+)", str(x).strip(), re.I)
    return f"HMDB{int(m.group(1)):07d}" if m else None


def kegg(x):
    x = str(x).strip().upper()
    return x if re.fullmatch(r"C\d{5}", x) else None


def build_html(structure, diagrams_path, model_path, id_translation, template, replacements=None):
    """The atlas page: template filled with the stage data, identifier lookup
    and diagrams. `replacements` fills extra placeholders (e.g. __METHOD__)."""
    d = json.loads(Path(structure).read_text(encoding="utf-8"))
    meta = d["nodes"]
    for m in meta:
        genes = m[3].split(";") if m[3] else []
        m[3] = ";".join(genes[:8]) + (f";+{len(genes) - 8} more" if len(genes) > 8 else "")

    # per-stage edges, as positions into each stage's node list
    model = load_sbml(model_path)
    meta_index = {m[0]: k for k, m in enumerate(meta)}
    for st, (key, _, _, rules, max_size, drop_obj, atom_pairs) in zip(d["stages"], stages(model, d.get("stage_set", "pruning"))):
        assert st["key"] == key
        g = build_graph(model, rules, max_size, drop_obj, atom_pairs=atom_pairs)
        assert [meta_index[x] for x in g.node_ids] == st["node"], f"node order differs in stage {key}"
        st["edges"] = g.edges.ravel().tolist()

    # identifiers: Human-GEM annotations + bridges through id_translation
    own = {}
    for s in model.species.values():
        r = own.setdefault(s.met_id, {"KEGG": set(), "HMDB": set(), "CHEBI": set()})
        r["KEGG"] |= {k for k in map(kegg, s.xrefs.get("kegg", [])) if k}
        r["HMDB"] |= {h for h in map(hmdb, s.xrefs.get("hmdb", [])) if h}
        r["CHEBI"] |= {c.upper() if c.upper().startswith("CHEBI:") else f"CHEBI:{c}" for c in s.xrefs.get("chebi", [])}
    names = [f"NAME{i}" for i in range(1, 6)]
    tr = pd.read_csv(id_translation, usecols=["KEGG", "HMDB"] + names, dtype=str,
                     encoding="utf-8", encoding_errors="replace")
    tr["KEGG"] = tr["KEGG"].map(lambda x: kegg(x) if isinstance(x, str) else None)
    tr["HMDB"] = tr["HMDB"].map(lambda x: hmdb(x) if isinstance(x, str) else None)
    kegg_rows = {k: g for k, g in tr.dropna(subset=["KEGG"]).groupby("KEGG")}
    hmdb_rows = {h: g for h, g in tr.dropna(subset=["HMDB"]).groupby("HMDB")}
    ids = []
    for m in meta:
        if m[2] != "m":
            ids.append(None)
            continue
        r = own.get(m[0], {"KEGG": set(), "HMDB": set(), "CHEBI": set()})
        entry = [[x, "model"] for x in sorted(r["KEGG"] | r["HMDB"] | r["CHEBI"])]
        rows = [kegg_rows[k] for k in r["KEGG"] if k in kegg_rows] + [hmdb_rows[h] for h in r["HMDB"] if h in hmdb_rows]
        bridged, syn = set(), set()
        for rows_df in rows:
            bridged |= set(rows_df["KEGG"].dropna()) | set(rows_df["HMDB"].dropna())
            for c in names:
                syn |= {s.strip() for s in rows_df[c].dropna() if 2 < len(s.strip()) < 60}
        entry += [[x, "id_translation"] for x in sorted(bridged - r["KEGG"] - r["HMDB"])]
        ids.append([entry, sorted({s for s in syn if s.lower() != m[1].lower()})[:8]])

    diagrams = json.loads(Path(diagrams_path).read_text(encoding="utf-8"))
    assert [s["key"] for s in diagrams["stages"]] == [s["key"] for s in d["stages"]], "diagram stages out of order"
    for n in diagrams["nodes"]:
        genes = n[3].split(";") if n[3] else []
        n[3] = ";".join(genes[:8]) + (f";+{len(genes) - 8} more" if len(genes) > 8 else "")

    payload = json.dumps({"nodes": meta, "ids": ids, "stages": d["stages"], "diagrams": diagrams},
                         separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = (Path(template).read_text(encoding="utf-8")
            .replace("__MODEL__", d["model"].split()[-1])
            .replace("__SAMPLES__", str(d["samples"])).replace("__PAIRS__", f"{d['samples'] ** 2:,}")
            .replace("__SEED__", str(d["seed"])))
    for k, v in (replacements or {}).items():
        html = html.replace(k, v)
    return html.replace("__DATA__", payload)          # last: the data may contain placeholder-like text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default=str(ROOT / "results/structure/structure.json"))
    ap.add_argument("--diagrams", default=str(ROOT / "results/stage_diagrams/diagrams.json"))
    ap.add_argument("--model", default=str(ROOT / "data/Human-GEM.xml"))
    ap.add_argument("--id-translation", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/pruning_atlas.html"))
    args = ap.parse_args()
    html = build_html(args.structure, args.diagrams, args.model, args.id_translation,
                      Path(__file__).with_name("atlas_template.html"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
