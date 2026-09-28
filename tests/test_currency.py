import unittest

from metaboenrich.gem import Model, Reaction, Species
from metaboenrich.network import build_graph, resolve_currency

METS = {
    # name: formula (Human-GEM style, charged forms)
    "ATP": "C10H12N5O13P3", "ADP": "C10H12N5O10P2", "AMP": "C10H12N5O7P",
    "H2O": "H2O", "Pi": "HO4P", "H+": "H",
    "glucose": "C6H12O6", "glucose-6-phosphate": "C6H11O9P",
    "adenylosuccinate": "C14H14N5O11P", "fumarate": "C4H2O4",
    "CoA": "C21H32N7O16P3S", "acetyl-CoA": "C23H34N7O17P3S", "acetate": "C2H3O2",
    "dephospho-CoA": "C21H33N7O13P2S",
    "adenosine": "C10H13N5O4", "cytidine": "C9H13N3O5", "NH3": "H3N",
    "UTP": "C9H11N2O15P3", "CTP": "C9H12N3O14P3", "CMP": "C9H12N3O8P",
    "acetoacetyl-CoA": "C25H36N7O18P3S",
}
RXNS = {
    "HK": ({"ATP", "glucose"}, {"ADP", "glucose-6-phosphate", "H+"}),
    "ADSL": ({"adenylosuccinate"}, {"AMP", "fumarate"}),
    "ACS": ({"acetate", "CoA", "ATP"}, {"acetyl-CoA", "AMP", "Pi"}),
    "DPCK": ({"dephospho-CoA", "ATP"}, {"CoA", "ADP"}),
    "ADK": ({"ATP", "adenosine"}, {"ADP", "AMP"}),                  # adenosine kinase
    "CTPS": ({"UTP", "ATP", "NH3"}, {"CTP", "ADP", "Pi"}),          # CTP synthase
    "UCK": ({"ATP", "cytidine"}, {"ADP", "CMP"}),                   # cytidine kinase
    "THL": ({"acetoacetyl-CoA", "CoA"}, {"acetyl-CoA"}),            # thiolase (2 acetyl-CoA)
}


def toy_model():
    ids = {name: f"MAM{i:05d}" for i, name in enumerate(METS)}
    species = {f"M_{ids[n]}c": Species(f"M_{ids[n]}c", ids[n], n, "c", {}, f) for n, f in METS.items()}
    sid = {n: f"M_{ids[n]}c" for n in METS}
    reactions = {
        rid: Reaction(rid, rid, {sid[m]: 1 for m in subs}, {sid[m]: 1 for m in prods},
                      True, False, [])
        for rid, (subs, prods) in RXNS.items()
    }
    return Model("toy", "0", species, reactions, {}), ids


class RoleRuleTest(unittest.TestCase):
    def setUp(self):
        self.model, self.ids = toy_model()
        rules, _ = resolve_currency(self.model, mode="role")
        self.graph = build_graph(self.model, rules)
        self.name = {v: k for k, v in self.ids.items()}
        self.decisions = {(self.name[m], r): d for m, r, d, _ in self.graph.currency_audit}

    def neighbours(self, rid):
        r = self.graph.index[rid]
        return {self.graph.node_names[m] for m, rr in self.graph.edges if rr == r}

    def test_exchange_pairs_are_removed(self):
        self.assertEqual(self.decisions[("ATP", "HK")], "exchange")          # ATP -> ADP
        self.assertEqual(self.decisions[("ADP", "HK")], "exchange")
        self.assertEqual(self.decisions[("CoA", "ACS")], "exchange")         # CoA -> acetyl-CoA (loaded)
        self.assertEqual(self.neighbours("HK"), {"glucose", "glucose-6-phosphate"})

    def test_synthesis_is_kept(self):
        self.assertEqual(self.decisions[("AMP", "ADSL")], "kept")            # adenylosuccinate -> AMP
        self.assertEqual(self.decisions[("CoA", "DPCK")], "kept")            # dephospho-CoA -> CoA
        self.assertEqual(self.decisions[("ATP", "DPCK")], "exchange")        # ATP -> ADP still removed
        self.assertEqual(self.neighbours("DPCK"), {"dephospho-CoA", "CoA"})
        self.assertIn("AMP", self.neighbours("ADSL"))

    def test_pairing_is_one_to_one(self):
        # ATP pairs with ADP (closest formula); the AMP made from adenosine is synthesis.
        self.assertEqual(self.decisions[("ATP", "ADK")], "exchange")
        self.assertEqual(self.decisions[("ADP", "ADK")], "exchange")
        self.assertEqual(self.decisions[("AMP", "ADK")], "kept")
        self.assertEqual(self.neighbours("ADK"), {"adenosine", "AMP"})

    def test_other_nucleotides_are_not_loaded_carriers(self):
        # ATP's formula contains CMP's, but ATP is not a loaded cytosine carrier.
        self.assertEqual(self.decisions[("CTP", "CTPS")], "kept")
        self.assertEqual(self.decisions[("UTP", "CTPS")], "kept")
        self.assertEqual(self.neighbours("CTPS"), {"UTP", "CTP"})
        self.assertEqual(self.decisions[("CMP", "UCK")], "kept")
        self.assertEqual(self.neighbours("UCK"), {"cytidine", "CMP"})

    def test_loaded_carrier_can_partner_twice(self):
        # Thiolase makes 2 acetyl-CoA; stored as a set, one acetyl-CoA must partner CoA.
        self.assertEqual(self.decisions[("CoA", "THL")], "exchange")
        self.assertEqual(self.neighbours("THL"), {"acetoacetyl-CoA", "acetyl-CoA"})

    def test_inorganic_removed_everywhere(self):
        self.assertNotIn("Pi", self.graph.node_names)
        self.assertNotIn("H+", self.graph.node_names)

    def test_remove_mode_drops_all_cofactors(self):
        rules, _ = resolve_currency(self.model, mode="remove")
        g = build_graph(self.model, rules)
        self.assertFalse({"ATP", "ADP", "AMP", "CoA"} & set(g.node_names))


if __name__ == "__main__":
    unittest.main()
