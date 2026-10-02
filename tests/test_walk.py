import unittest

import numpy as np

from metaboenrich.flow import current_flow
from metaboenrich.network import Graph
from metaboenrich.walk import ANY, absorbing_walk


def rxn_graph(n_mets, reactions):
    """reactions: [(substrates, products, forward, backward)]; metabolites 0..n_mets-1,
    reaction i is node n_mets + i."""
    ids = [f"m{i}" for i in range(n_mets)] + [f"r{i}" for i in range(len(reactions))]
    edges, sides, arcs, rev = [], [], [], {}
    for i, (subs, prods, f, b) in enumerate(reactions):
        r = n_mets + i
        rev[r] = f and b
        for m in subs:
            edges.append((m, r)); sides.append("substrate")
            arcs += [(m, r)] * f + [(r, m)] * b
        for m in prods:
            edges.append((m, r)); sides.append("product")
            arcs += [(r, m)] * f + [(m, r)] * b
    return Graph(node_ids=ids, node_names=ids,
                 is_reaction=np.array([False] * n_mets + [True] * len(reactions)),
                 edges=np.array(edges), arcs=np.array(arcs, dtype=np.int64).reshape(-1, 2),
                 reaction_genes={}, index={v: i for i, v in enumerate(ids)}, edge_side=sides, reversible=rev)


IRR, REV = (True, False), (True, True)


class AbsorbingWalkTest(unittest.TestCase):
    def test_chain_carries_everything(self):
        # m0 -> r0 -> m1 -> r1 -> m2
        g = rxn_graph(3, [([0], [1], *IRR), ([1], [2], *IRR)])
        res = absorbing_walk(g, [0], [2], {0: 1.0, 2: 1.0})
        np.testing.assert_allclose(res.conductivity[[1, 3, 4]], 1.0)
        np.testing.assert_allclose(res.net_forward[[3, 4]], 1.0)
        self.assertEqual(res.pairs, [(0, 2)])
        self.assertAlmostEqual(res.endpoint[0]["eff_conductance"], 1.0)   # every walker arrives

    def test_direction_is_respected(self):
        # r0 reversible, r1 runs only backward (m2 -> m1): no route from m0 to m2.
        g = rxn_graph(3, [([0], [1], *REV), ([1], [2], False, True)])
        res = absorbing_walk(g, [0], [2], {0: 1.0, 2: 1.0})
        self.assertEqual(res.pairs, [])
        self.assertEqual(res.excluded, [(0, 2, "no directed path to a target")])
        np.testing.assert_allclose(res.conductivity, 0.0)
        # ... and from m2 to m0 the walk runs both reactions backward.
        back = absorbing_walk(g, [2], [0], {0: 1.0, 2: 1.0})
        np.testing.assert_allclose(back.conductivity[[1, 3, 4]], 1.0)
        np.testing.assert_allclose(back.net_forward[[3, 4]], -1.0)         # both used backward

    def test_no_cosubstrate_hop(self):
        # r0: m0 + m1 -> m2. Current flow passes m0 -r0- m1; the walk cannot.
        g = rxn_graph(3, [([0, 1], [2], *IRR)])
        self.assertAlmostEqual(current_flow(g, [0], [1]).conductivity[3], 1.0)
        self.assertEqual(absorbing_walk(g, [0], [1], {0: 1.0, 1: 1.0}).pairs, [])

    def test_no_immediate_reversal(self):
        # r0: m0 + m1 <-> m2 (reversible). Forward to m2 then straight back to m1 is
        # forbidden, so m1 is unreachable from m0 unless another reaction leads there.
        g = rxn_graph(3, [([0, 1], [2], *REV)])
        self.assertEqual(absorbing_walk(g, [0], [1], {0: 1.0, 1: 1.0}).pairs, [])
        # With a second, separate route m2 -> r1 -> m1 the walk goes r0 forward, then r1.
        g2 = rxn_graph(3, [([0, 1], [2], *REV), ([2], [1], *IRR)])
        res = absorbing_walk(g2, [0], [1], {0: 1.0, 1: 1.0})
        np.testing.assert_allclose(res.conductivity[[2, 3, 4]], 1.0)
        np.testing.assert_allclose(res.net_forward[[3, 4]], 1.0)

    def test_leaked_walks_are_not_scored(self):
        # m0 -> r0 -> m1 (target) and m0 -> r1 -> m2 (dead end): half the walkers leak.
        g = rxn_graph(3, [([0], [1], *IRR), ([0], [2], *IRR)])
        res = absorbing_walk(g, [0], [1], {0: 1.0, 1: 1.0})
        self.assertAlmostEqual(res.endpoint[0]["eff_conductance"], 0.5)   # reach probability
        np.testing.assert_allclose(res.conductivity[[3, 4, 2]], [1.0, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(res.walk_info["walk_success_fraction"], 0.5)

    def test_partial_absorption(self):
        # m0 -> r0 -> m1 (target) -> r1 -> m2 (target). Full absorption stops every
        # walker at m1; with kappa a share exp(-kappa * phi) walks on to m2.
        g = rxn_graph(3, [([0], [1], *IRR), ([1], [2], *IRR)])
        phi = {0: 1.0, 1: 1.0, 2: 2.0}
        full = absorbing_walk(g, [0], [1, 2], phi)
        self.assertEqual(full.pairs, [(0, 1)])
        self.assertIn((0, 2, "no walk reaches this target"), full.excluded)
        res = absorbing_walk(g, [0], [1, 2], phi, kappa=1.0)
        a1, a2 = 1 - np.exp(-1.0), 1 - np.exp(-2.0)
        success = a1 + (1 - a1) * a2                                        # the rest leaks past m2
        self.assertAlmostEqual(res.endpoint[1]["eff_conductance"], a1 / success)
        self.assertAlmostEqual(res.endpoint[2]["eff_conductance"], (1 - a1) * a2 / success)
        self.assertAlmostEqual(res.conductivity[4], (1 - a1) * a2 / success)   # r1
        self.assertAlmostEqual(res.conductivity[1], (1 - a1) * a2 / success)   # m1 relays what it passes on

    def test_fold_change_weights_sources(self):
        # Two sources with their own reaction into one target; phi 3 vs 1.
        g = rxn_graph(3, [([0], [2], *IRR), ([1], [2], *IRR)])
        res = absorbing_walk(g, [0, 1], [2], {0: 3.0, 1: 1.0, 2: 1.0})
        np.testing.assert_allclose(res.conductivity[[3, 4]], [0.75, 0.25])
        np.testing.assert_allclose(res.bottleneck[[3, 4]], 1.0)

    def test_one_sided_ground(self):
        # Only a source: the measured, unchanged metabolite m2 absorbs.
        g = rxn_graph(3, [([0], [1], *IRR), ([1], [2], *IRR)])
        res = absorbing_walk(g, [0], [], {0: 1.0}, ground=[2])
        np.testing.assert_allclose(res.conductivity[[3, 4]], 1.0)
        self.assertEqual(res.pairs, [(0, ANY)])

    def test_matches_monte_carlo(self):
        # Branches, a reversible reaction, a directed cycle, a dead end, partial
        # absorption, and a target walkers can pass through.
        reactions = [([0], [1], *IRR),        # r0
                     ([1, 2], [3], *REV),     # r1
                     ([3], [4], *IRR),        # r2  -> m4 target
                     ([1], [5], *IRR),        # r3  -> m5 dead end
                     ([3], [1], *IRR),        # r4  cycle m1 -> m3 -> m1
                     ([2], [6], *IRR),        # r5  -> m6 target
                     ([6], [4], *IRR)]        # r6  past m6 to m4
        n_mets = 7
        g = rxn_graph(n_mets, reactions)
        sources, targets = [0, 2], [4, 6]
        phi, kappa = {0: 2.0, 2: 1.0, 4: 1.0, 6: 0.5}, 1.0
        res = absorbing_walk(g, sources, targets, phi, kappa=kappa)

        # Independent simulation, straight from the reaction list.
        choices = {m: [] for m in range(n_mets)}             # metabolite -> [(rxn, dir, ins, outs)]
        for i, (subs, prods, f, b) in enumerate(reactions):
            for d, ins, outs, ok in ((1, subs, prods, f), (-1, prods, subs, b)):
                if ok:
                    for m in ins:
                        choices[m].append((i, d, outs))
        edge_of = {(int(m), int(r)): e for e, (m, r) in enumerate(g.edges)}
        absorb = {t: 1 - np.exp(-kappa * phi[t]) for t in targets}
        rng = np.random.default_rng(1)
        B = np.abs(g.incidence().toarray())
        T_all, w_all, reach = [], [], {}
        for s in sources:
            net, absorbed, ok, walks = np.zeros(len(g.edges)), np.zeros(len(targets)), 0, 40000
            for _ in range(walks):
                m, came, path = s, None, np.zeros(len(g.edges))
                while True:
                    if came is not None and m in absorb and rng.random() < absorb[m]:
                        ok += 1
                        net += path
                        absorbed[targets.index(m)] += 1
                        break
                    opts = [c for c in choices[m] if came is None or (c[0], c[1]) != (came[0], -came[1])]
                    if not opts:
                        break                                    # dead end: leaks
                    i, d, outs = opts[rng.integers(len(opts))]
                    nxt = outs[rng.integers(len(outs))]
                    path[edge_of[(m, n_mets + i)]] += 1
                    path[edge_of[(nxt, n_mets + i)]] -= 1
                    m, came = nxt, (i, d)
            net /= ok
            T = 0.5 * B.T @ np.abs(net)
            T[targets] -= 0.5 * absorbed / ok
            T[s] = 0.0
            T_all.append(np.maximum(T, 0))
            w_all.append(phi[s] * ok / walks)
            reach[s] = ok / walks
        w = np.array(w_all) / sum(w_all)
        expected = np.array(T_all).T @ w
        np.testing.assert_allclose(res.conductivity, expected, atol=0.02)
        for s in sources:
            self.assertAlmostEqual(res.endpoint[s]["eff_conductance"], reach[s], delta=0.01)


if __name__ == "__main__":
    unittest.main()
