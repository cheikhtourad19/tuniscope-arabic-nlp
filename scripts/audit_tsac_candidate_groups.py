"""Audit a conservative, train-only TSAC grouping rule without writing data.

Run from the project root: python3 scripts/audit_tsac_candidate_groups.py
Candidate links constrain future splitting; they do not assert equal meaning.
"""

from collections import Counter, defaultdict
import json

from screen_tsac_within_train import ROOT, UNION_LEDGER, retained_forms, sha256


INDEX = ROOT / "data/interim/tsac_within_train_near_2026-10-16.json"


def load_groups():
    with INDEX.open(encoding="utf-8") as stream:
        index = json.load(stream)
    if index["exclusion_ledger_sha256"] != sha256(UNION_LEDGER):
        raise ValueError("Candidate index does not match the exclusion ledger")
    if index["schema_version"] != "1.0":
        raise ValueError("Unsupported candidate index")

    forms = retained_forms()
    refs = {(item["file"], item["line"]): item["label"] for item in forms.values()}
    if len(refs) != 11827:
        raise ValueError("Retained references are not unique")
    parent = {ref: ref for ref in refs}

    def find(ref):
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    pairs = index["primary_pairs"] + index["short_pairs"]
    if len(index["primary_pairs"]) != 203 or len(index["short_pairs"]) != 171:
        raise ValueError("Candidate pair counts changed")
    seen = set()
    for pair in pairs:
        a = (pair["file_a"], pair["line_a"])
        b = (pair["file_b"], pair["line_b"])
        if a not in refs or b not in refs or a == b:
            raise ValueError("Candidate pair is not two retained train rows")
        if refs[a] != pair["label_a"] or refs[b] != pair["label_b"]:
            raise ValueError("Candidate labels do not match retained train rows")
        edge = tuple(sorted((a, b)))
        if edge in seen:
            raise ValueError("Repeated candidate edge")
        seen.add(edge)
        parent[find(a)] = find(b)

    groups = defaultdict(list)
    for ref in refs:
        groups[find(ref)].append(ref)
    return refs, [tuple(sorted(group)) for group in groups.values()], len(seen)


def main():
    refs, groups, edge_count = load_groups()
    multi = [group for group in groups if len(group) > 1]
    mixed = [group for group in multi if len({refs[ref] for ref in group}) > 1]
    label_counts = Counter(refs.values())
    summary = {
        "rule_version": "tsac-candidate-components-v1",
        "rule": "Every primary or short candidate edge joins two retained train rows; use connected components only as indivisible split groups",
        "candidate_edges": edge_count,
        "retained_train_rows": len(refs),
        "retained_label_counts": dict(sorted(label_counts.items())),
        "groups_total": len(groups),
        "multi_row_groups": len(multi),
        "rows_in_multi_row_groups": sum(map(len, multi)),
        "largest_group": max(map(len, groups)),
        "mixed_label_groups": len(mixed),
        "mixed_label_rows": sum(map(len, mixed)),
        "source_test_rows_examined": 0,
        "dataset_or_manifest_written": False,
    }
    if (summary["groups_total"], summary["multi_row_groups"],
            summary["rows_in_multi_row_groups"], summary["mixed_label_groups"]) != (11484, 209, 552, 7):
        raise ValueError("Grouping audit differs from the previous screen")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
