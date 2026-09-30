"""Carbon-skeleton pairs per reaction, from atom mapping.

For each reaction, find which substrate-product pairs share carbon atoms,
i.e. where part of a molecule's skeleton is carried into another. A group
transfer (phosphate, amino group, electrons) shares no carbon, so e.g. in a
transaminase (glutamate + pyruvate -> 2-oxoglutarate + alanine) the pairs are
glutamate/2-oxoglutarate and pyruvate/alanine, not glutamate/alanine.

Atom maps come from RXNMapper (Schwaller et al., Sci Adv 2021) run on
Human-GEM's own SMILES (model/metabolites.tsv of the pinned release).
Reactions whose participants lack structures, or that the mapper cannot
handle, get no pairs and are left intact by the graph builder (flagged).

    python -m metaboenrich.atoms --out data/atom_pairs.json
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .gem import HUMAN_GEM_VERSION, load_sbml

METABOLITES_URL = (f"https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/{HUMAN_GEM_VERSION}"
                   "/model/metabolites.tsv")
METABOLITES_SHA256 = "c31439f997a6d4a68ba6d00971243a600d1c9523b736d876b285a8f8a4c9ab34"


def load_smiles(path="data/Human-GEM_metabolites.tsv"):
    """{met_id (compartment-free): SMILES} from Human-GEM's metabolites.tsv
    (downloaded from the pinned release if missing)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(METABOLITES_URL, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != METABOLITES_SHA256:
            tmp.unlink()
            raise SystemExit("Downloaded metabolites.tsv has an unexpected SHA-256")
        tmp.replace(path)
    t = pd.read_csv(path, sep="\t", dtype=str)
    t = t[t.metSmiles.notna() & (t.metSmiles.str.len() > 0)]
    return dict(zip(t.metsNoComp, t.metSmiles))


def reaction_sides(model, rxn):
    """Compartment-free substrate and product met_ids, minus metabolites on
    both sides (transport)."""
    subs = {model.species[s].met_id for s in rxn.substrates}
    prods = {model.species[s].met_id for s in rxn.products}
    both = subs & prods
    return sorted(subs - both), sorted(prods - both)


def _canon(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def carbon_pairs(mapped_rxn, subs, prods, smiles):
    """Shared-carbon counts {(substrate, product): n} from an atom-mapped
    reaction SMILES. Molecules are matched back to met_ids by canonical SMILES."""
    from rdkit import Chem
    left, right = mapped_rxn.split(">>")
    canon_to_met = {}
    for m in subs + prods:
        c = _canon(smiles[m])
        if c:
            canon_to_met.setdefault(c, []).append(m)

    def frags(side, allowed):
        out = []
        for f in side.split("."):
            mol = Chem.MolFromSmiles(f)
            if mol is None:
                continue
            c = _canon(f)
            met = next((x for x in canon_to_met.get(c, []) if x in allowed), None)
            out.append((met, mol))
        return out

    owner = {}                                   # map number -> substrate met
    for met, mol in frags(left, set(subs)):
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() and a.GetSymbol() == "C":
                owner[a.GetAtomMapNum()] = met
    pairs = {}
    for met, mol in frags(right, set(prods)):
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C" and a.GetAtomMapNum() in owner:
                s = owner[a.GetAtomMapNum()]
                if s and met:
                    pairs[(s, met)] = pairs.get((s, met), 0) + 1
    return pairs


def map_model(model, smiles, batch_size=8, limit=None, log_every=200):
    """{reaction id: {"pairs": [[sub, prod, n_carbons], ...], "confidence": c}}
    or {"error": reason}."""
    from rxnmapper import RXNMapper
    mapper = RXNMapper()
    todo, out = [], {}
    for rxn in model.reactions.values():
        subs, prods = reaction_sides(model, rxn)
        if len(set(subs) | set(prods)) < 2:
            continue
        missing = [m for m in subs + prods if m not in smiles]
        if missing:
            out[rxn.id] = {"error": f"no SMILES for {len(missing)} participant(s)"}
            continue
        rsmi = ".".join(smiles[m] for m in subs) + ">>" + ".".join(smiles[m] for m in prods)
        todo.append((rxn.id, subs, prods, rsmi))
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        try:
            results = mapper.get_attention_guided_atom_maps([c[3] for c in chunk])
        except Exception:                       # one bad reaction spoils the batch: retry singly
            results = []
            for c in chunk:
                try:
                    results.append(mapper.get_attention_guided_atom_maps([c[3]])[0])
                except Exception as err:
                    results.append({"error": f"mapper: {type(err).__name__}"})
        for (rid, subs, prods, _), res in zip(chunk, results):
            if "error" in res:
                out[rid] = res
                continue
            try:
                pairs = carbon_pairs(res["mapped_rxn"], subs, prods, smiles)
                out[rid] = {"pairs": [[a, b, n] for (a, b), n in sorted(pairs.items())],
                            "confidence": round(float(res["confidence"]), 4)}
            except Exception as err:
                out[rid] = {"error": f"parse: {type(err).__name__}"}
        done = i + len(chunk)
        if log_every and (done // log_every != (done - len(chunk)) // log_every or done == len(todo)):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  mapped {done}/{len(todo)} reactions ({rate:.1f}/s; ~{(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min left)")
    return out


def reaction_kegg_ids(sbml_path):
    """{SBML reaction id: [KEGG reaction id, ...]} from the model's annotations."""
    import xml.etree.ElementTree as ET
    ns = {"sbml": "http://www.sbml.org/sbml/level3/version1/core",
          "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "bqbiol": "http://biomodels.net/biology-qualifiers/"}
    res = "{%s}resource" % ns["rdf"]
    out = {}
    for el in ET.parse(sbml_path).getroot().iterfind(".//sbml:reaction", ns):
        ks = [li.get(res).rsplit("/", 1)[1] for li in el.iterfind(".//bqbiol:is/rdf:Bag/rdf:li", ns)
              if "kegg.reaction/" in (li.get(res) or "")]
        if ks:
            out[el.get("id")] = ks
    return out


def fetch_kegg_rclass_pairs(kegg_ids, cache="data/kegg_rclass_pairs.json", pause=0.35):
    """{KEGG reaction: [[C1, C2], ...]} from each KEGG REACTION entry's RCLASS
    field (the curated structure-sharing compound pairs; successor of RPAIR).
    Cached, so KEGG is only queried for ids not already fetched."""
    cache = Path(cache)
    known = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted(set(kegg_ids) - set(known))
    for i in range(0, len(todo), 10):
        batch = todo[i:i + 10]
        url = "https://rest.kegg.jp/get/" + "+".join(f"rn:{k}" for k in batch)
        text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
        for entry in text.split("///"):
            lines = entry.strip("\n").splitlines()
            if not lines or not lines[0].startswith("ENTRY"):
                continue
            rid = lines[0].split()[1]
            pairs, in_rc = [], False
            for ln in lines:
                if ln.startswith("RCLASS"):
                    in_rc = True
                elif ln[:1].strip():            # next field
                    in_rc = False
                if in_rc:
                    pairs += [p.split("_") for p in ln[12:].split() if "_" in p and p.startswith("C")]
            known[rid] = pairs
        for k in batch:
            known.setdefault(k, [])
        cache.write_text(json.dumps(known))
        time.sleep(pause)
    return {k: known.get(k, []) for k in kegg_ids}


def kegg_pairs_for_model(model, sbml_path="data/Human-GEM.xml", cache="data/kegg_rclass_pairs.json"):
    """{reaction id: [[substrate, product], ...]} from KEGG RCLASS, translated
    to Human-GEM metabolites through their KEGG compound ids. Reactions whose
    KEGG entry gives no pair that maps onto their participants are omitted."""
    rk = reaction_kegg_ids(sbml_path)
    kp = fetch_kegg_rclass_pairs(sorted({k for v in rk.values() for k in v}), cache)
    kegg_of = {}
    for s in model.species.values():
        for k in s.xrefs.get("kegg", []):
            kegg_of.setdefault(s.met_id, set()).add(k)
    out = {}
    for rid, ks in rk.items():
        if rid not in model.reactions:
            continue
        subs, prods = reaction_sides(model, model.reactions[rid])
        pairs = set()
        for k in ks:
            for c1, c2 in kp.get(k, []):
                for a in subs:
                    for b in prods:
                        ka, kb = kegg_of.get(a, set()), kegg_of.get(b, set())
                        if (c1 in ka and c2 in kb) or (c2 in ka and c1 in kb):
                            pairs.add((a, b))
        if pairs:
            out[rid] = [list(p) for p in sorted(pairs)]
    return out


def clean_pairs(pairs):
    """Keep a mapped pair if the substrate is the product's main carbon source
    or the product is the substrate's main carbon destination.

    Chosen against KEGG RCLASS pairs on the 2,027 Human-GEM reactions that
    have both (non-currency pairs): precision 90.0%, recall 96.3%, 88.3% of
    reactions exactly right (raw mapper output: 85.4% / 98.3% / 84.2%; on
    multi-substrate reactions 61.4% vs 37.7% exactly right). Raw maps invent
    cross-pairs in transaminases (AKG -> OAA, aspartate -> glutamate)."""
    best_p, best_s = {}, {}
    for a, b, n in pairs:
        best_p[b] = max(best_p.get(b, 0), n)
        best_s[a] = max(best_s.get(a, 0), n)
    return [[a, b, n] for a, b, n in pairs if n == best_p[b] or n == best_s[a]]


def _rule_top_donor(pairs):
    best = {}
    for a, b, n in pairs:
        best[b] = max(best.get(b, 0), n)
    return [[a, b, n] for a, b, n in pairs if n == best[b]]


CLEANING_RULES = {
    "Raw atom maps (any shared carbon)": lambda p: p,
    "Main carbon source of each product": _rule_top_donor,
    "Main source or main destination (used)": clean_pairs,
}


def evaluate_rules(model, raw_mapped, kegg, excluded):
    """Agreement of each cleaning rule with KEGG RCLASS pairs, on reactions that
    have both, counting only pairs between metabolites not in `excluded`
    (the currency metabolites, whose links the graph never uses).
    Returns rows: rule, reaction set, n, precision, recall, exactly right."""
    common = [r for r in kegg if r in raw_mapped and raw_mapped[r].get("pairs")]
    multi = [r for r in common if len([x for x in reaction_sides(model, model.reactions[r])[0] if x not in excluded]) >= 2]
    rows = []
    for label, rule in CLEANING_RULES.items():
        for set_name, sel in (("all reactions", common), ("multi-substrate reactions", multi)):
            tp = fp = fn = exact = 0
            for r in sel:
                pred = {(a, b) for a, b, _ in rule(raw_mapped[r]["pairs"]) if a not in excluded and b not in excluded}
                gold = {tuple(p) for p in kegg[r] if p[0] not in excluded and p[1] not in excluded}
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
                exact += pred == gold
            rows.append({"rule": label, "reactions": set_name, "n": len(sel),
                         "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                         "exactly_right": exact / max(len(sel), 1)})
    return rows


def combine_pairs(raw_mapped, kegg):
    """{reaction id: {"pairs": [[sub, prod], ...], "source": "kegg" | "rxnmapper"}}:
    KEGG RCLASS pairs where available, otherwise cleaned RXNMapper pairs."""
    out = {}
    for rid, v in raw_mapped.items():
        if "pairs" in v and v["pairs"]:
            out[rid] = {"pairs": [p[:2] for p in clean_pairs(v["pairs"])], "source": "rxnmapper"}
    for rid, pairs in kegg.items():
        out[rid] = {"pairs": [list(p) for p in pairs], "source": "kegg"}
    return out


def load_atom_pairs(path):
    """{reaction id: [(substrate, product), ...]} for build_graph(atom_pairs=...)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {rid: [tuple(p) for p in v["pairs"]] for rid, v in data.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="metaboenrich.atoms", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="data/Human-GEM.xml")
    ap.add_argument("--raw", default="data/atom_pairs_raw.json",
                    help="RXNMapper output (computed if missing; ~10 min on CPU)")
    ap.add_argument("--out", default="data/atom_pairs.json")
    ap.add_argument("--limit", type=int, default=None, help="map only the first N reactions (for testing)")
    args = ap.parse_args(argv)
    model = load_sbml(args.model)
    raw_path = Path(args.raw)
    if raw_path.exists() and not args.limit:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        raw = map_model(model, load_smiles(), limit=args.limit)
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
    kegg = kegg_pairs_for_model(model, args.model)
    combined = combine_pairs(raw, kegg)
    Path(args.out).write_text(json.dumps(combined), encoding="utf-8")
    src = pd.Series([v["source"] for v in combined.values()]).value_counts().to_dict()
    print(f"Wrote {args.out}: {len(combined)} reactions with carbon pairs {src}; "
          f"{sum('pairs' not in v for v in raw.values())} could not be mapped")


if __name__ == "__main__":
    main()
