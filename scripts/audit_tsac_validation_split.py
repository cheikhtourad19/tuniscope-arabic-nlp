"""Simulate a reproducible grouped TSAC train/validation split; write nothing.

Run: python3 scripts/audit_tsac_validation_split.py
Only projected, retained TSAC training rows are considered. The official test
is not opened. Output contains aggregate counts and a split fingerprint only.
"""

from collections import Counter, defaultdict
import hashlib
import json
import random

from audit_tsac_candidate_groups import load_groups


SEED = 42
VALIDATION_FRACTION = 0.15


def split_groups(refs, groups):
    """Sample 15% of multi-row groups, then fill per-label targets with singletons."""
    rng = random.Random(SEED)
    multi = sorted((group for group in groups if len(group) > 1))
    singletons = sorted((group for group in groups if len(group) == 1))
    rng.shuffle(multi)
    target = {label: round(count * VALIDATION_FRACTION)
              for label, count in Counter(refs.values()).items()}
    validation = set()
    counts = Counter()
    selected_multi = 0
    for group in multi[:round(len(multi) * VALIDATION_FRACTION)]:
        group_counts = Counter(refs[ref] for ref in group)
        if all(counts[label] + number <= target[label]
               for label, number in group_counts.items()):
            validation.update(group)
            counts.update(group_counts)
            selected_multi += 1

    by_label = defaultdict(list)
    for group in singletons:
        by_label[refs[group[0]]].append(group[0])
    for label in sorted(target):
        rng.shuffle(by_label[label])
        needed = target[label] - counts[label]
        if needed > len(by_label[label]):
            raise ValueError("Not enough singletons to reach validation target")
        validation.update(by_label[label][:needed])
        counts[label] += needed

    if counts != Counter(target):
        raise ValueError("Validation balance target was not reached")
    if any(0 < len(validation.intersection(group)) < len(group) for group in groups):
        raise ValueError("A candidate family crosses train and validation")
    if len(validation) != sum(target.values()):
        raise ValueError("Validation count mismatch")
    return validation, target, selected_multi


def main():
    refs, groups, edge_count = load_groups()
    validation, target, selected_multi = split_groups(refs, groups)
    train = set(refs) - validation
    train_counts = Counter(refs[ref] for ref in train)
    val_counts = Counter(refs[ref] for ref in validation)
    mixed = [group for group in groups if len({refs[ref] for ref in group}) > 1]
    fingerprint = hashlib.sha256()
    for ref in sorted(refs):
        fingerprint.update(f"{ref[0]}:{ref[1]}:{'validation' if ref in validation else 'train'}\n".encode())
    summary = {
        "split_rule_version": "tsac-grouped-validation-v1",
        "group_rule_version": "tsac-candidate-components-v1",
        "seed": SEED,
        "validation_fraction_target": VALIDATION_FRACTION,
        "validation_label_targets": dict(sorted(target.items())),
        "train_rows": len(train),
        "train_labels": dict(sorted(train_counts.items())),
        "validation_rows": len(validation),
        "validation_labels": dict(sorted(val_counts.items())),
        "validation_fraction_actual": round(len(validation) / len(refs), 6),
        "candidate_edges": edge_count,
        "groups_total": len(groups),
        "multi_row_groups_selected_for_validation": selected_multi,
        "mixed_label_groups_in_validation": sum(set(group) <= validation for group in mixed),
        "groups_crossing_train_validation": 0,
        "assignment_sha256": fingerprint.hexdigest(),
        "official_test_rows_examined": 0,
        "manifest_or_dataset_written": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
