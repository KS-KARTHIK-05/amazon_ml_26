"""Ground-truth edge parsing and deterministic positive-family components.

A "family" is a connected component of the bipartite graph whose nodes are S1
entities and S2/S3 target entities, linked by ground-truth match edges. Two S1
entities that share a matched target (a target with multiple owners) belong to
the same family and must never be split across train/tuning/holdout — see
agent.md invariant 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            return x
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def nodes(self):
        return list(self._parent.keys())


@dataclass
class TruthEdge:
    s1_entity_id: str
    target_entity_id: str
    target_source: str  # "S2" or "S3"


@dataclass
class TruthParseResult:
    edges: list[TruthEdge] = field(default_factory=list)
    s1_truth: dict[str, set[str]] = field(default_factory=dict)  # includes empty sets
    errors: list[str] = field(default_factory=list)
    target_owner_count: dict[str, int] = field(default_factory=dict)


def _target_source(target_entity_id: str) -> str | None:
    if target_entity_id.startswith("S2-"):
        return "S2"
    if target_entity_id.startswith("S3-"):
        return "S3"
    return None


def parse_ground_truth_chunk(
    chunk_rows: list[tuple[str, str]], result: TruthParseResult
) -> None:
    """Parse ``(source1_entity_id, matched_entity_ids)`` row tuples into
    ``result`` in place, so callers can stream large files chunk by chunk."""
    for s1_id, matched_field in chunk_rows:
        if s1_id in result.s1_truth:
            result.errors.append(f"duplicate source1_entity_id in ground truth: {s1_id}")
            continue
        targets: set[str] = set()
        matched_field = matched_field.strip() if matched_field else ""
        if matched_field:
            for target_id in matched_field.split(","):
                target_id = target_id.strip()
                if not target_id:
                    continue
                if target_id == s1_id or target_id.startswith("S1-"):
                    result.errors.append(
                        f"{s1_id}: ground truth references an S1 id as a target: {target_id}"
                    )
                    continue
                src = _target_source(target_id)
                if src is None:
                    result.errors.append(
                        f"{s1_id}: matched id has no S2-/S3- prefix: {target_id}"
                    )
                    continue
                if target_id in targets:
                    result.errors.append(
                        f"{s1_id}: duplicate target id in its own match list: {target_id}"
                    )
                    continue
                targets.add(target_id)
                result.edges.append(TruthEdge(s1_id, target_id, src))
                result.target_owner_count[target_id] = (
                    result.target_owner_count.get(target_id, 0) + 1
                )
        result.s1_truth[s1_id] = targets


def build_positive_families(result: TruthParseResult) -> dict[str, str]:
    """Return ``{s1_entity_id: family_id}``.

    Every S1 entity gets a family, including singletons (own component). Two
    S1 entities sharing any matched target land in the same family. The
    family_id is the union-find root over S1 ids only, chosen deterministically
    as the lexicographically smallest S1 id in the component so it is stable
    across re-runs regardless of edge insertion order.
    """
    uf = UnionFind()
    for s1_id in result.s1_truth:
        uf.find(s1_id)  # ensure isolated nodes exist

    target_to_s1: dict[str, list[str]] = {}
    for edge in result.edges:
        target_to_s1.setdefault(edge.target_entity_id, []).append(edge.s1_entity_id)
    for s1_ids in target_to_s1.values():
        first = s1_ids[0]
        for other in s1_ids[1:]:
            uf.union(first, other)

    root_to_members: dict[str, list[str]] = {}
    for s1_id in result.s1_truth:
        root = uf.find(s1_id)
        root_to_members.setdefault(root, []).append(s1_id)

    family_id_of_root = {
        root: min(members) for root, members in root_to_members.items()
    }
    return {
        s1_id: family_id_of_root[uf.find(s1_id)] for s1_id in result.s1_truth
    }


def multi_owner_targets(result: TruthParseResult) -> dict[str, int]:
    """Targets (S2/S3 ids) claimed by more than one S1 entity."""
    return {t: c for t, c in result.target_owner_count.items() if c > 1}
