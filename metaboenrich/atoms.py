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


def combine_pairs(raw_mapped, kegg, mode="kegg-first"):
    """{reaction id: {"pairs": [[sub, prod], ...], "source": ...}}.

    mode "kegg-first": KEGG RCLASS pairs where KEGG has the reaction, otherwise
    cleaned RXNMapper pairs. mode "union" (used by 3-rule pruning): every pair
    from either source, sorted, so minor carbon donors KEGG leaves out
    (carbamoyl-phosphate -> citrulline) are kept."""
    if mode == "union":
        out = {}
        for rid in sorted(set(raw_mapped) | set(kegg)):
            v = raw_mapped.get(rid, {})
            pairs = {tuple(p[:2]) for p in clean_pairs(v["pairs"])} if v.get("pairs") else set()
            pairs |= {tuple(p) for p in kegg.get(rid, [])}
            if pairs:
                out[rid] = {"pairs": [list(p) for p in sorted(pairs)], "source": "union"}
        return out
    out = {}
    for rid, v in raw_mapped.items():
        if "pairs" in v and v["pairs"]:
            out[rid] = {"pairs": [p[:2] for p in clean_pairs(v["pairs"])], "source": "rxnmapper"}
    for rid, pairs in kegg.items():
        out[rid] = {"pairs": [list(p) for p in pairs], "source": "kegg"}
    return out


COUPLE_MIN_REACTIONS = 10


def carbon_method_pairs(model, pairs, couple_min_reactions=COUPLE_MIN_REACTIONS, keep_rarest=False,
                        no_rescue=frozenset(), log=None, merge=None, carriers=None):
    """The standalone carbon method, with no hand-made currency lists.

    Rule 1: keep a substrate-product pair only if both molecules contain
    carbon (KEGG's curated pairs also include non-carbon pairs such as
    H2O2/O2). Water, protons, phosphate, O2, NH3 and ions therefore never link.
    Rule 2: a pair that recurs in `couple_min_reactions` or more reactions is a
    cofactor couple doing bookkeeping (ATP/ADP, NAD+/NADH, CoA/acetyl-CoA,
    glutamate/2-oxoglutarate, UDP/UDP-sugar ...) and is removed everywhere.

    Keep-rarest safeguard (keep_rarest=True): a reaction whose carbon pairs
    are all couples would lose every link and drop out with its genes
    (glutamate dehydrogenase: glutamate/2-oxoglutarate and NAD+/NADH). There,
    the couple found in the fewest reactions is kept, ties broken by
    metabolite ids. Reactions in `no_rescue` (transport-coupled ones, whose
    only pair is usually ATP/ADP) are never rescued. Each rescue is appended
    to `log` (a list) when one is given. With keep_rarest=False the output
    is exactly that of the two rules alone.

    carriers: {(met, met) sorted: carrier} from find_carriers (Rule 3); these
    carrier / loaded-form pairs are removed everywhere, like couples.

    merge: {met_id: merged id} (acyl_coa_merge). Pairs are renamed first and
    pairs within one merged node dropped, so couples are counted on the
    merged network (CoA / long-chain acyl-CoA becomes one pair).

    Returns ({reaction id: [(sub, prod), ...]}, {(met, met): n_reactions} of
    the couples removed)."""
    from .network import parse_formula
    carbon = {}
    for s in model.species.values():
        carbon.setdefault(s.met_id, parse_formula(s.formula)["C"] > 0)
    if merge:
        carbon.update({v: True for v in merge.values()})
        pairs = {rid: list(dict.fromkeys((merge.get(a, a), merge.get(b, b)) for a, b, *_ in ps
                                         if merge.get(a, a) != merge.get(b, b)))
                 for rid, ps in pairs.items()}
    rule1 = {rid: [(a, b) for a, b in ps if carbon.get(a) and carbon.get(b)] for rid, ps in pairs.items()}
    freq = {}
    for ps in rule1.values():
        for a, b in {tuple(sorted(p)) for p in ps}:
            freq[(a, b)] = freq.get((a, b), 0) + 1
    couples = {p: n for p, n in freq.items() if n >= couple_min_reactions}
    drop = set(couples) | set(carriers or ())
    out = {rid: [(a, b) for a, b in ps if tuple(sorted((a, b))) not in drop] for rid, ps in rule1.items()}
    if keep_rarest:
        for rid, ps in rule1.items():
            if out[rid] or not ps or rid in no_rescue:
                continue
            candidates = sorted({tuple(sorted(p)) for p in ps})
            best = min(candidates, key=lambda p: (freq[p], p))
            out[rid] = [(a, b) for a, b in ps if tuple(sorted((a, b))) == best][:1]
            if log is not None:
                log.append({"reaction": rid, "kept_pair": best, "kept_pair_count": freq[best],
                            "candidates": [(p, freq[p]) for p in candidates], "mapped": True})
    return out, couples


def carbon_rule1_pairs(model, pairs):
    """Rule 1 only (see carbon_method_pairs): pairs whose partners both contain carbon."""
    out, _ = carbon_method_pairs(model, pairs, couple_min_reactions=10 ** 9)
    return out


def carbon_method_rules(model, pairs, couples, pools=frozenset(), keep_rarest=False, no_rescue=frozenset(),
                        log=None, merge=None, carriers=None):
    """CurrencyRules applying the two rules to reactions with no atom map,
    from formulas alone: metabolites without carbon are removed everywhere
    (Rule 1), and in an unmapped reaction a detected couple is removed when
    one partner is a substrate and the other a product (Rule 2). Mapped
    reactions are filtered by their pairs in build_graph; these rules only
    remove what their pairs already exclude there.

    keep_rarest, no_rescue, log: the keep-rarest safeguard of
    carbon_method_pairs, for unmapped reactions. Where removing couples would
    leave fewer than two metabolites (so the reaction drops out), the rarest
    couple present across the two sides is kept. Couple counts are the
    network-wide counts from the mapped reactions (`couples`), not recomputed.

    carriers: {carrier: {loaded forms}} from find_carriers (Rule 3). In an
    unmapped reaction the carrier itself is removed when a loaded form of it
    is on the other side; the loaded form keeps its links."""
    from .network import CurrencyRules, parse_formula
    removed = {}
    for s in model.species.values():
        if parse_formula(s.formula)["C"] == 0:
            removed[s.met_id] = "no carbon"
    partner = {}
    for a, b in couples:
        partner.setdefault(a, set()).add(b)
        partner.setdefault(b, set()).add(a)
    edits = {}
    for rid, rxn in model.reactions.items():
        if rid in pairs:
            continue
        subs = {model.species[s].met_id for s in rxn.substrates}
        prods = {model.species[s].met_id for s in rxn.products}
        if merge:
            subs = {merge.get(m, m) for m in subs}
            prods = {merge.get(m, m) for m in prods}
        hit = {m for m in subs if partner.get(m, set()) & prods}
        hit |= {m for m in prods if partner.get(m, set()) & subs}
        if carriers:
            hit |= {c for c in subs if carriers.get(c, set()) & prods}
            hit |= {c for c in prods if carriers.get(c, set()) & subs}
        if hit and keep_rarest and rid not in no_rescue:
            both = subs & prods
            if len((subs | prods) - both - set(removed) - hit) < 2:
                present = sorted({tuple(sorted((a, b))) for a in subs - both for b in prods - both
                                  if tuple(sorted((a, b))) in couples})
                if present:
                    best = min(present, key=lambda p: (couples[p], p))
                    hit -= set(best)
                    if log is not None:
                        log.append({"reaction": rid, "kept_pair": best, "kept_pair_count": couples[best],
                                    "candidates": [(p, couples[p]) for p in present], "mapped": False})
        if hit:
            edits[rid] = hit
    return CurrencyRules(removed=removed, family_of={}, members={}, loaded_core={}, formulas={},
                         pools=set(pools), edits=edits, names={}, loaded_tokens={})


def transport_reactions(model, carbon_cargo=True):
    """Reactions carrying the same metabolite on both sides before compartments
    are merged (transporters). carbon_cargo=True: only where that metabolite
    contains carbon, so proton-pumping enzymes (respiratory complexes,
    transhydrogenase) and ion pumps are not counted as transporters."""
    from .network import parse_formula
    carbon = {s.met_id: parse_formula(s.formula)["C"] > 0 for s in model.species.values()}
    out = set()
    for rid, rxn in model.reactions.items():
        shared = {model.species[s].met_id for s in rxn.substrates} & {model.species[s].met_id for s in rxn.products}
        if shared and (not carbon_cargo or any(carbon[m] for m in shared)):
            out.add(rid)
    return out


ACYL_COA_CLASSES = (  # (min acyl carbons, max, merged id, name); chain-length classes of fatty-acid biochemistry
    (13, 21, "ACYLCOA_LC", "long-chain acyl-CoA (C13-C21)"),
    (22, 999, "ACYLCOA_VLC", "very-long-chain acyl-CoA (C22+)"),
)
COA_CARBONS = 21


def acyl_coa_merge(model, smiles_path="data/Human-GEM_metabolites.tsv", classes=ACYL_COA_CLASSES):
    """Merge long acyl-CoAs into one node per chain-length class.

    An acyl-CoA is found from its structure (Human-GEM SMILES): it contains the
    CoA pantetheine-ADP substructure and a thioester. Its acyl chain length is
    its carbon count minus CoA's 21. Short (C2-C5) and medium (C6-C12) acyl-CoAs,
    including acetyl- and malonyl-CoA, stay individual; pool pseudo-metabolites
    are never merged. Returns ({met_id: merged id}, {merged id: name},
    {met_id: acyl carbons} for every acyl-CoA found)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    coa = Chem.MolFromSmarts("SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OC")
    thio = Chem.MolFromSmarts("[CX3](=O)S")
    names = {s.met_id: s.name for s in model.species.values()}
    lengths, merge = {}, {}
    for mid, smi in load_smiles(smiles_path).items():
        if mid not in names or "pool" in names[mid].lower():
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not (mol.HasSubstructMatch(coa) and mol.HasSubstructMatch(thio)):
            continue
        n = sum(a.GetSymbol() == "C" for a in mol.GetAtoms()) - COA_CARBONS
        lengths[mid] = n
        for lo, hi, new, _ in classes:
            if lo <= n <= hi:
                merge[mid] = new
    return merge, {new: name for _, _, new, name in classes}, lengths


CARRIER_MIN_LOADED = 10


def _carrier_site(small, large):
    """Symmetry classes (atoms of `small`) where `large` attaches extra carbon to a
    whole copy of `small`, through heteroatom bonds only; None if `large` is
    not `small` loaded that way. A bond between two carbons means the carbon
    skeleton itself was extended (thiolase, ALA synthase), not a carrier loaded."""
    from rdkit import Chem

    def n_c(mol):
        return sum(a.GetSymbol() == "C" for a in mol.GetAtoms())
    if not (large.GetNumHeavyAtoms() > small.GetNumHeavyAtoms() and n_c(large) > n_c(small)):
        return None
    cls = list(Chem.CanonicalRankAtoms(small, breakTies=False))
    for match in large.GetSubstructMatches(small, uniquify=False, maxMatches=200):
        pos = {a: k for k, a in enumerate(match)}
        sites, ok = set(), True
        for b in large.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (i in pos) != (j in pos):
                if large.GetAtomWithIdx(i).GetSymbol() == "C" and large.GetAtomWithIdx(j).GetSymbol() == "C":
                    ok = False
                    break
                sites.add(cls[pos[i] if i in pos else pos[j]])
        if ok and sites:
            return frozenset(sites)
    return None


def find_carriers(model, pairs, smiles_path="data/Human-GEM_metabolites.tsv", min_loaded=CARRIER_MIN_LOADED):
    """Rule 3: carriers, from structure.

    A carrier is a molecule found whole inside at least `min_loaded` of its
    carbon pair partners, every one of them attached at the same atom of the
    carrier and through a heteroatom (thioester, ester, amide): CoA in
    acyl-CoAs, L-carnitine in acylcarnitines, glutathione in its conjugates.
    The pair carrier / loaded form is loading or unloading, not a skeleton
    change, and is removed everywhere (carbon_method_pairs, carriers=);
    pairs between two loaded forms (acetyl-CoA -> acetoacetyl-CoA) are kept.

    Returns ({carrier: {loaded forms}}, {(met, met) sorted: carrier} for the
    pairs to remove)."""
    from collections import Counter
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    smiles = load_smiles(smiles_path)
    uncharge, mols = rdMolStandardize.Uncharger(), {}

    def mol(x):
        if x not in mols:
            m = Chem.MolFromSmiles(smiles[x]) if x in smiles else None
            if m is not None:
                m = uncharge.uncharge(m)
                Chem.RemoveStereochemistry(m)
            mols[x] = m
        return mols[x]
    loaded = {}
    for a, b in sorted({tuple(sorted(p[:2])) for ps in carbon_rule1_pairs(model, pairs).values() for p in ps}):
        ma, mb = mol(a), mol(b)
        if ma is None or mb is None:
            continue
        for small, large, ms, ml in ((a, b, ma, mb), (b, a, mb, ma)):
            site = _carrier_site(ms, ml)
            if site:
                loaded.setdefault(small, {})[large] = site
    carriers = {c: set(d) for c, d in loaded.items()
                if len(d) >= min_loaded and len(Counter(d.values())) == 1}
    remove = {tuple(sorted((c, x))): c for c, xs in carriers.items() for x in xs}
    return carriers, remove


def pool_metabolites(model):
    """Lumped pseudo-metabolites: every species Human-GEM names a pool
    ("acyl-CoA pool", "PI pool", "fatty acid-VLDL pool" ...)."""
    return {s.met_id for s in model.species.values() if "pool" in s.name.lower()}


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
    ap.add_argument("--combine", choices=["kegg-first", "union"], default="kegg-first",
                    help="kegg-first: KEGG pairs where available, else atom maps (data/atom_pairs.json); "
                         "union: both, as used by 3-rule pruning (write to data/atom_pairs_union.json)")
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
    combined = combine_pairs(raw, kegg, args.combine)
    Path(args.out).write_text(json.dumps(combined), encoding="utf-8")
    src = pd.Series([v["source"] for v in combined.values()]).value_counts().to_dict()
    print(f"Wrote {args.out}: {len(combined)} reactions with carbon pairs {src}; "
          f"{sum('pairs' not in v for v in raw.values())} could not be mapped")


if __name__ == "__main__":
    main()
