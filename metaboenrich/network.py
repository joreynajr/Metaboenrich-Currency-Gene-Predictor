"""Build the metabolite-reaction graph (protocol §1.1 / §2.1) and handle
currency metabolites.

Currency metabolites participate in hundreds to thousands of reactions and
would otherwise short-circuit the network. Two kinds are distinguished:

* Inorganic currency (H+, H2O, Pi, ions, ...) and generic carriers
  ([protein], ferredoxins, cytochrome b5, thioredoxins) are removed from
  every reaction. Reactions containing a lumped pool metabolite (biomass
  pools, "cofactors and vitamins", "xenobiotics", ...) are dropped.
* Cofactor currency (ATP, NAD+, CoA, SAM, ...) is removed only where it acts
  as an exchanger in a cycle, i.e. where a partner form of the same cofactor
  is on the other side of the reaction (ATP -> ADP, NAD+ -> NADH,
  CoA -> acyl-CoA, UDP-glucose -> UDP). Where it is made from, or turned into,
  something outside its family (adenylosuccinate -> AMP, dephospho-CoA -> CoA,
  NAD+ -> nicotinamide + ADP-ribose) the edge is kept, so the cofactor stays
  in the network through its own synthesis and catabolism.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Removed from every reaction: no meaningful synthesis/catabolism of their own.
INORGANIC_CURRENCY = [
    "H+", "H2O", "O2", "CO2", "HCO3-", "H2O2", "NH3", "Pi", "PPi",
    "Na+", "K+", "chloride", "sulfate", "Fe2+", "Fe3+",
]

# Generic carriers removed from every reaction: pure exchangers with no
# synthesis in the model, or placeholder nodes such as the generic [protein],
# which otherwise links every amino acid through "[protein] to
# [protein]-L-X conversion" bookkeeping reactions.
CARRIER_CURRENCY = [
    "[protein]",
    "oxidized ferredoxin", "reduced ferredoxin",
    "oxidized adrenal ferredoxin", "reduced adrenal ferredoxin",
    "ferricytochrome B5", "ferrocytochrome B5",
    "thioredoxin", "oxidized thioredoxin", "mitothioredoxin", "mitooxidized thioredoxin",
]

# Lumped pool metabolites: any reaction containing one is bookkeeping
# (biomass assembly, "X to vitamin A derivatives conversion", ...) and is dropped.
POOL_METABOLITES = [
    "biomass", "cofactor_pool_biomass", "lipid_pool_biomass", "metabolite_pool_biomass",
    "protein_pool_biomass", "cofactors and vitamins", "xenobiotics", "steroids",
    "arachidonate derivatives", "vitamin A derivatives", "vitamin D derivatives",
    "vitamin E derivatives", "lipid droplet", "Gm4-Pool",
]

# Cofactor families (Human-GEM names). `loaded` / `tokens`: a metabolite is a
# loaded form of the carrier (acyl-CoAs, UDP-sugars, CDP-alcohols,
# GDP-sugars, CMP-sialic acids) and counts as a family member if its name
# contains one of the tokens as a word (UDP-glucose, acetyl-CoA; not dUDP)
# and its formula contains the `loaded` member's formula plus extra carbon.
# The name test stops other nucleotides (ATP contains every atom of CMP)
# from passing as loaded carriers.
COFACTOR_FAMILIES = {
    "adenine nucleotides": {"members": ["ATP", "ADP", "AMP"]},
    "guanine nucleotides": {"members": ["GTP", "GDP", "GMP"], "loaded": "GDP", "tokens": ["GDP"]},
    "uracil nucleotides": {"members": ["UTP", "UDP", "UMP"], "loaded": "UDP", "tokens": ["UDP"]},
    "cytosine nucleotides": {"members": ["CTP", "CDP", "CMP"], "loaded": "CMP", "tokens": ["CDP", "CMP"]},
    "NAD": {"members": ["NAD+", "NADH"]},
    "NADP": {"members": ["NADP+", "NADPH"]},
    "FAD": {"members": ["FAD", "FADH2"]},
    "CoA": {"members": ["CoA"], "loaded": "CoA", "tokens": ["CoA"]},
    "SAM": {"members": ["SAM", "SAH"]},
    "PAPS": {"members": ["PAPS", "PAP"]},
    "ubiquinone": {"members": ["ubiquinone", "ubiquinol"]},
    "PLP": {"members": ["pyridoxal-phosphate", "pyridoxamine-phosphate"]},
}

# Kept for reference / --currency-mode remove: every default currency metabolite.
DEFAULT_CURRENCY_NAMES = INORGANIC_CURRENCY + CARRIER_CURRENCY + [
    m for fam in COFACTOR_FAMILIES.values() for m in fam["members"]
]


DEFAULT_REACTION_EDITS = Path(__file__).with_name("reaction_edits.tsv")


def load_reaction_edits(path, model):
    """Read a reaction-edits table (reaction_id, met_id, ...; tab-separated).
    Returns {reaction id: {met_id, ...}} and a list of rows that don't match the model."""
    edits, unknown = {}, []
    met_ids = {s.met_id for s in model.species.values()}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            rid, met = row["reaction_id"], row["met_id"]
            if rid not in model.reactions or met not in met_ids:
                unknown.append((rid, met))
            else:
                edits.setdefault(rid, set()).add(met)
    return edits, unknown


def parse_formula(formula):
    """'C21H32N7O16P3S' -> Counter of heavy atoms (H ignored; 'R' kept)."""
    counts = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        if el != "H":
            counts[el] += int(n) if n else 1
    return counts


@dataclass
class CurrencyRules:
    removed: dict                  # met_id -> reason; dropped from every reaction
    family_of: dict                # met_id -> family name, for role-filtered cofactors
    members: dict                  # family -> set of explicit member met_ids
    loaded_core: dict              # family -> heavy-atom Counter of the carrier core
    formulas: dict = field(default_factory=dict)   # met_id -> heavy-atom Counter
    pools: set = field(default_factory=set)        # met_ids whose reactions are dropped
    edits: dict = field(default_factory=dict)      # reaction id -> met_ids to remove from it
    names: dict = field(default_factory=dict)      # met_id -> name
    loaded_tokens: dict = field(default_factory=dict)  # family -> compiled name pattern

    def in_family(self, family, met):
        if met in self.members[family]:
            return True
        core, pattern = self.loaded_core.get(family), self.loaded_tokens.get(family)
        if core is None or pattern is None or any(met in ms for ms in self.members.values()):
            return False
        f = self.formulas.get(met)
        return (bool(f) and bool(pattern.search(self.names.get(met, "")))
                and all(f[el] >= n for el, n in core.items()) and f["C"] > core["C"])

    def classify(self, subs, prods):
        """{cofactor met_id: exchange partner or None} for the role-filtered
        cofactors in a reaction.

        Within each family, members on the two sides are paired, closest
        formulas first (ATP pairs with ADP rather than AMP). A paired cofactor
        is an exchanger; an unpaired one is made from, or turned into,
        something outside its family, so it is kept. Cofactors pair
        one-to-one. Loaded carriers (acyl-CoAs, UDP-sugars) may partner more
        than one cofactor, because reactions are stored as sets and lose
        stoichiometry (thiolase: acetoacetyl-CoA + CoA -> 2 acetyl-CoA). They
        are ordinary metabolites, so their own edges are never removed.
        """
        present = (subs | prods) & self.family_of.keys()
        result = {}
        for fam in sorted({self.family_of[m] for m in present}):
            left = sorted(m for m in subs if self.in_family(fam, m))
            right = sorted(m for m in prods if self.in_family(fam, m))
            pairs = sorted((self._distance(a, b), a, b) for a in left for b in right)
            partner = {}
            reusable = lambda m: m not in self.family_of
            for _, a, b in pairs:
                if (a in partner and not reusable(a)) or (b in partner and not reusable(b)):
                    continue
                if a in partner and b in partner:
                    continue
                partner.setdefault(a, b)
                partner.setdefault(b, a)
            for m in present:
                if self.family_of[m] == fam:
                    result[m] = partner.get(m)
        return result

    def _distance(self, a, b):
        fa, fb = self.formulas.get(a, Counter()), self.formulas.get(b, Counter())
        return sum(abs(fa[el] - fb[el]) for el in set(fa) | set(fb))


def resolve_currency(model, mode="role", use_defaults=True, extra_ids=(), degree_cutoff=None,
                     max_reaction_size=20):
    """Build the currency rules.

    mode 'role':   inorganic currency removed everywhere; cofactors removed only
                   where they act as exchangers (see module docstring).
    mode 'remove': every default currency metabolite removed everywhere.
    Returns (rules, names not found in the model).
    """
    by_name = {}
    formulas = {}
    for s in model.species.values():
        by_name.setdefault(s.name, s.met_id)
        formulas.setdefault(s.met_id, parse_formula(s.formula))

    removed, family_of, members, loaded_core, loaded_tokens, missing = {}, {}, {}, {}, {}, []
    pools = set()
    met_names = {s.met_id: s.name for s in model.species.values()}
    if use_defaults:
        for names, reason in ((INORGANIC_CURRENCY, "inorganic"), (CARRIER_CURRENCY, "generic carrier")):
            for name in names:
                if name in by_name:
                    removed[by_name[name]] = reason
                else:
                    missing.append(name)
        for name in POOL_METABOLITES:
            if name in by_name:
                pools.add(by_name[name])
            else:
                missing.append(name)
        for fam, spec in COFACTOR_FAMILIES.items():
            ids = []
            for name in spec["members"]:
                if name in by_name:
                    ids.append(by_name[name])
                else:
                    missing.append(name)
            members[fam] = set(ids)
            if "loaded" in spec and spec["loaded"] in by_name:
                loaded_core[fam] = formulas[by_name[spec["loaded"]]]
                loaded_tokens[fam] = re.compile(
                    r"(?<![A-Za-z0-9])(?:" + "|".join(map(re.escape, spec["tokens"])) + r")(?![A-Za-z])",
                    re.IGNORECASE)  # Human-GEM writes both "UDP-glucose" and "udp-ribose"
            for met in ids:
                if mode == "remove":
                    removed[met] = f"cofactor ({fam})"
                else:
                    family_of[met] = fam
    for met in extra_ids:
        removed[met] = "user supplied"
        family_of.pop(met, None)
    if degree_cutoff is not None:
        deg, _ = metabolite_degrees(model, max_reaction_size)
        for met, d in deg.items():
            if d > degree_cutoff and met not in removed and met not in family_of:
                removed[met] = f"degree {d} > {degree_cutoff}"
    return CurrencyRules(removed, family_of, members, loaded_core, formulas, pools,
                         names=met_names, loaded_tokens=loaded_tokens), missing


@dataclass
class Graph:
    node_ids: list          # met_id (metabolite nodes) or reaction id
    node_names: list
    is_reaction: np.ndarray  # bool per node
    edges: np.ndarray        # (m, 2) int array, [metabolite, reaction]
    # Directed arcs for §2.1: (u, v) pairs, metabolite->reaction or reaction->metabolite
    arcs: np.ndarray
    reaction_genes: dict     # reaction node index -> [ensembl ids]
    index: dict              # node id -> node index
    # (met_id, reaction id, "kept" | "exchange", partner met_id or "") for
    # every role-filtered cofactor occurrence in a reaction that made it into the graph
    currency_audit: list = field(default_factory=list)
    edge_side: list = field(default_factory=list)    # per edge: "substrate" | "product"
    reversible: dict = field(default_factory=dict)   # reaction node index -> bool

    @property
    def n(self):
        return len(self.node_ids)

    def incidence(self):
        """Signed edge-node incidence matrix B (m x n); Laplacian L = B^T B."""
        m = len(self.edges)
        rows = np.repeat(np.arange(m), 2)
        cols = self.edges.ravel()
        vals = np.tile([1.0, -1.0], m)
        return sp.csr_matrix((vals, (rows, cols)), shape=(m, self.n))

    def components(self):
        """Connected-component label per node (undirected)."""
        u, v = self.edges.T
        adj = sp.csr_matrix((np.ones(len(u)), (u, v)), shape=(self.n, self.n))
        _, labels = connected_components(adj, directed=False)
        return labels

    def reachable_from(self, start):
        """Nodes reachable from `start` following directed arcs (for §2.2)."""
        if not hasattr(self, "_succ"):
            self._succ = [[] for _ in range(self.n)]
            for u, v in self.arcs:
                self._succ[u].append(v)
        seen = np.zeros(self.n, dtype=bool)
        seen[start] = True
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in self._succ[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
        return seen


def metabolite_degrees(model, max_reaction_size):
    """Reactions per metabolite (compartments collapsed, no currency handling)."""
    deg = Counter()
    for _, subs, prods, _ in _usable_reactions(model, max_reaction_size, rules=None):
        for met in subs | prods:
            deg[met] += 1
    names = {s.met_id: s.name for s in model.species.values()}
    return deg, names


def _usable_reactions(model, max_reaction_size, rules, drop_objective=True):
    """Yield (reaction, substrate met_ids, product met_ids, audit rows) after
    collapsing compartments and applying the currency rules.

    Dropped: the objective (biomass) reaction, blocked reactions, reactions
    containing a lumped pool metabolite (when rules are given), reactions
    with more than `max_reaction_size` metabolites (pool/biomass-like
    pseudo-reactions; None disables the cap), and reactions left with < 2
    distinct metabolites (transport, exchange, or reactions made only of
    currency metabolites).
    """
    for rxn in model.reactions.values():
        if (drop_objective and rxn.is_objective) or not (rxn.forward or rxn.backward):
            continue
        subs = {model.species[s].met_id for s in rxn.substrates}
        prods = {model.species[s].met_id for s in rxn.products}
        if rules is not None:
            if (subs | prods) & rules.pools:
                continue
            edited = rules.edits.get(rxn.id, set())
            subs, prods = subs - edited, prods - edited
        both = subs & prods  # same metabolite on both sides, e.g. transport
        subs, prods = subs - both, prods - both
        audit = []
        if rules is not None:
            drop = set(rules.removed)
            for met, partner in sorted(rules.classify(subs, prods).items()):
                if partner:
                    drop.add(met)
                audit.append((met, rxn.id, "exchange" if partner else "kept", partner or ""))
            subs, prods = subs - drop, prods - drop
        mets = subs | prods
        if len(mets) < 2 or (max_reaction_size is not None and len(mets) > max_reaction_size):
            continue
        yield rxn, subs, prods, audit


def build_graph(model, rules, max_reaction_size=20, drop_objective=True):
    met_names = {s.met_id: s.name for s in model.species.values()}
    node_ids, node_names, is_rxn, index = [], [], [], {}

    def node(nid, name, rxn):
        if nid not in index:
            index[nid] = len(node_ids)
            node_ids.append(nid)
            node_names.append(name)
            is_rxn.append(rxn)
        return index[nid]

    edges, arcs, reaction_genes, audit_rows, sides, reversible = [], [], {}, [], [], {}
    for rxn, subs, prods, audit in _usable_reactions(model, max_reaction_size, rules, drop_objective):
        audit_rows.extend(audit)
        r = node(rxn.id, rxn.name, True)
        reaction_genes[r] = rxn.genes
        reversible[r] = rxn.forward and rxn.backward
        for met in sorted(subs):
            m = node(met, met_names[met], False)
            edges.append((m, r))
            sides.append("substrate")
            if rxn.forward:
                arcs.append((m, r))
            if rxn.backward:
                arcs.append((r, m))
        for met in sorted(prods):
            m = node(met, met_names[met], False)
            edges.append((m, r))
            sides.append("product")
            if rxn.forward:
                arcs.append((r, m))
            if rxn.backward:
                arcs.append((m, r))

    return Graph(
        node_ids=node_ids,
        node_names=node_names,
        is_reaction=np.array(is_rxn, dtype=bool),
        edges=np.array(edges, dtype=np.int64),
        arcs=np.array(arcs, dtype=np.int64),
        reaction_genes=reaction_genes,
        index=index,
        currency_audit=audit_rows,
        edge_side=sides,
        reversible=reversible,
    )
