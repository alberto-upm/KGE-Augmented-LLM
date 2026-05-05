"""División del grafo por bloques completos.

Primero se aparta el `test_system` (10% de bloques completos, sin filtraciones)
para evaluación end-to-end. Sobre el resto se hace el split interno del KGE
a nivel de tripleta (train/val/test_kge).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .io_graph import Triple


@dataclass
class Splits:
    train: list[Triple]
    val: list[Triple]
    test_kge: list[Triple]
    test_system: dict[str, list[Triple]]   # block_id → tripletas


def split_by_block(
    blocks: dict[str, list[Triple]],
    extra_triples: list[Triple],
    test_system_ratio: float,
    kge_val_ratio: float,
    kge_test_ratio: float,
    seed: int,
) -> Splits:
    rng = random.Random(seed)

    block_ids = list(blocks.keys())
    rng.shuffle(block_ids)

    n_test_system = max(1, int(len(block_ids) * test_system_ratio))
    test_system_ids = set(block_ids[:n_test_system])
    train_block_ids = block_ids[n_test_system:]

    test_system = {bid: blocks[bid] for bid in test_system_ids}

    pool: list[Triple] = []
    for bid in train_block_ids:
        pool.extend(blocks[bid])
    pool.extend(extra_triples)

    rng.shuffle(pool)
    n = len(pool)
    n_val = int(n * kge_val_ratio)
    n_test = int(n * kge_test_ratio)
    val = pool[:n_val]
    test_kge = pool[n_val : n_val + n_test]
    train = pool[n_val + n_test :]

    return Splits(train=train, val=val, test_kge=test_kge, test_system=test_system)


def write_tsv(triples: list[Triple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")


def write_test_system(test_system: dict[str, list[Triple]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for bid, triples in test_system.items():
            for s, p, o in triples:
                f.write(f"{bid}\t{s}\t{p}\t{o}\n")


def run_split(cfg: dict, blocks: dict[str, list[Triple]], extra_triples: list[Triple]) -> Splits:
    s = cfg["split"]
    splits = split_by_block(
        blocks=blocks,
        extra_triples=extra_triples,
        test_system_ratio=s["test_system_ratio"],
        kge_val_ratio=s["kge_val_ratio"],
        kge_test_ratio=s["kge_test_ratio"],
        seed=s["seed"],
    )
    out_dir = Path(s["out_dir"])
    write_tsv(splits.train, out_dir / "train.tsv")
    write_tsv(splits.val, out_dir / "val.tsv")
    write_tsv(splits.test_kge, out_dir / "test_kge.tsv")
    write_test_system(splits.test_system, out_dir / "test_system.tsv")
    return splits
