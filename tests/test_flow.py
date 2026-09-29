import unittest

import numpy as np

from metaboenrich.flow import current_flow
from metaboenrich.network import Graph


def toy_graph(n_mets, n_rxns, edges, arcs=()):
    ids = [f"m{i}" for i in range(n_mets)] + [f"r{i}" for i in range(n_rxns)]
    return Graph(
        node_ids=ids, node_names=ids,
        is_reaction=np.array([False] * n_mets + [True] * n_rxns),
        edges=np.array(edges), arcs=np.array(arcs, dtype=np.int64).reshape(-1, 2),
        reaction_genes={}, index={v: i for i, v in enumerate(ids)},
    )


class CurrentFlowTest(unittest.TestCase):
    def test_parallel_routes_split_current(self):
        # m0 -r0(3)- m1 and m0 -r1(4)- m1: two equal routes, half the current each.
        g = toy_graph(2, 2, [(0, 2), (1, 2), (0, 3), (1, 3)])
        res = current_flow(g, [0], [1])
        np.testing.assert_allclose(res.conductivity[[2, 3]], [0.5, 0.5])
        np.testing.assert_allclose(res.endpoint[0]["eff_conductance"], 1.0)  # R = 2 || 2 = 1

    def test_series_route_is_bottleneck(self):
        # m0 - r0 - m2 - r1 - m1, plus a dead-end branch m2 - r2 - m3.
        g = toy_graph(4, 3, [(0, 4), (2, 4), (2, 5), (1, 5), (2, 6), (3, 6)])
        res = current_flow(g, [0], [1])
        np.testing.assert_allclose(res.bottleneck[[4, 5, 2]], 1.0)   # all current passes here
        np.testing.assert_allclose(res.bottleneck[[6, 3]], 0.0, atol=1e-12)  # dead end carries none
        np.testing.assert_allclose(res.endpoint[0]["eff_conductance"], 0.25)  # 4 unit resistors in series

    def test_matches_pseudoinverse(self):
        rng = np.random.default_rng(0)
        mets, rxns = 8, 10
        edges = sorted({(int(rng.integers(mets)), mets + r) for r in range(rxns) for _ in range(3)})
        edges += [(m, mets + m % rxns) for m in range(mets)]  # ensure connectivity-ish
        g = toy_graph(mets, rxns, sorted(set(edges)))
        if len(set(g.components())) > 1:
            self.skipTest("random graph disconnected")
        B = g.incidence().toarray()
        Lp = np.linalg.pinv(B.T @ B)
        res = current_flow(g, [0], [1])
        b = np.zeros(g.n); b[0], b[1] = 1, -1
        x = Lp @ b
        T = 0.5 * np.abs(B).T @ np.abs(B @ x)
        T[[0, 1]] = 0
        np.testing.assert_allclose(res.conductivity, T, atol=1e-10)
        np.testing.assert_allclose(res.endpoint[0]["eff_conductance"], 1 / (x[0] - x[1]))

    def test_leave_one_out_discounts_single_metabolite_routes(self):
        # Sources m0, m1 each have one reaction (r0, r1) into hub m2; m2 -r2- m3 (target).
        # r0 and r1 are forced by one source each; r2 carries current from both.
        g = toy_graph(4, 3, [(0, 4), (2, 4), (1, 5), (2, 5), (2, 6), (3, 6)])
        res = current_flow(g, [0, 1], [3])
        r0, r1, r2 = 4, 5, 6
        np.testing.assert_allclose(res.bottleneck[[r0, r1, r2]], 1.0)
        np.testing.assert_allclose(res.bottleneck_loo[[r0, r1]], 0.0, atol=1e-12)
        np.testing.assert_allclose(res.bottleneck_loo[r2], 1.0)
        np.testing.assert_allclose(res.conductivity[[r0, r2]], [0.5, 1.0])
        np.testing.assert_allclose(res.conductivity_loo[[r0, r2]], [0.0, 1.0], atol=1e-12)

    def test_leave_one_out_matches_brute_force(self):
        rng = np.random.default_rng(1)
        mets, rxns = 10, 12
        edges = {(int(rng.integers(mets)), mets + r) for r in range(rxns) for _ in range(3)}
        edges |= {(m, mets + m % rxns) for m in range(mets)}
        g = toy_graph(mets, rxns, sorted(edges))
        if len(set(g.components())) > 1:
            self.skipTest("random graph disconnected")
        S, T = [0, 1, 2], [3, 4]
        res = current_flow(g, S, T)
        cond_min = np.full(g.n, np.inf)
        bott_min = np.full(g.n, np.inf)
        for side, drop in [(0, u) for u in S] + [(1, u) for u in T]:
            s_keep = [s for s in S if not (side == 0 and s == drop)]
            t_keep = [t for t in T if not (side == 1 and t == drop)]
            sub = current_flow(g, s_keep, t_keep)
            # Throughput of a DAA node excludes only its own pairs, so compare reactions.
            cond_min = np.minimum(cond_min, sub.conductivity)
            bott_min = np.minimum(bott_min, sub.bottleneck)
        rx = g.is_reaction
        np.testing.assert_allclose(res.conductivity_loo[rx], cond_min[rx], atol=1e-10)
        np.testing.assert_allclose(res.bottleneck_loo[rx], bott_min[rx], atol=1e-10)

    def test_one_sided_ground_splits_withdrawal(self):
        # m0 -r0- m1 -r1- m2; source m0, ground {m1, m2}: half the current is
        # withdrawn at m1, so r0 carries 1 and r1 carries 0.5.
        g = toy_graph(3, 2, [(0, 3), (1, 3), (1, 4), (2, 4)])
        res = current_flow(g, [0], [], ground=[1, 2])
        self.assertEqual(len(res.pairs), 1)
        np.testing.assert_allclose(res.conductivity[[3, 4]], [1.0, 0.5])
        np.testing.assert_allclose(res.conductivity[[1, 2]], 0.0)       # ground nodes are sinks

    def test_one_sided_matches_pseudoinverse(self):
        rng = np.random.default_rng(3)
        mets, rxns = 9, 11
        edges = {(int(rng.integers(mets)), mets + r) for r in range(rxns) for _ in range(3)}
        edges |= {(m, mets + m % rxns) for m in range(mets)}
        g = toy_graph(mets, rxns, sorted(edges))
        if len(set(g.components())) > 1:
            self.skipTest("random graph disconnected")
        S, G = [0, 1], [5, 6, 7]
        res = current_flow(g, S, [], ground=G)
        B = g.incidence().toarray()
        Lp = np.linalg.pinv(B.T @ B)
        T_sum = np.zeros(g.n)
        for s in S:
            b = np.zeros(g.n); b[s] = 1.0; b[G] -= 1.0 / len(G)
            x = Lp @ b
            T = 0.5 * np.abs(B).T @ np.abs(B @ x)
            T[s] = 0; T[G] = 0
            T_sum += T
        np.testing.assert_allclose(res.conductivity, T_sum / len(S), atol=1e-10)
        # targets-only data: current drawn at the target, supplied by the ground
        res_t = current_flow(g, [], [0], ground=G)
        self.assertEqual(len(res_t.pairs), 1)

    def test_directed_mode_excludes_unreachable_pairs(self):
        # m0 -> r0 -> m1 only (irreversible); a pair m1 -> m0 has no directed path.
        g = toy_graph(2, 1, [(0, 2), (1, 2)], arcs=[(0, 2), (2, 1)])
        self.assertEqual(len(current_flow(g, [1], [0], directed=True).pairs), 0)
        self.assertEqual(len(current_flow(g, [0], [1], directed=True).pairs), 1)
        self.assertEqual(len(current_flow(g, [1], [0], directed=False).pairs), 1)


if __name__ == "__main__":
    unittest.main()
