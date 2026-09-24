"""Train the visual skill head used after a grasp visibly fails."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from so100.action_head import SKILLS
from so100.encode import Siglip2
from so100.train_visual_action import CHECKPOINT, collect, pack
from so100.visual_action_head import RecoveryActionHead, VisualActionHead

RECOVERY_CHECKPOINT = CHECKPOINT.with_name("recovery.pt")
CACHE = Path("/tmp/so100-visual-action-slips-v2.pt")


def main() -> None:
    torch.manual_seed(2026)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if CACHE.exists():
        train_rows, val_rows = torch.load(CACHE, weights_only=False)
    else:
        eyes = Siglip2(device)
        train_rows, _ = collect(eyes, range(261000, 261064), slips=True)
        val_rows, _ = collect(eyes, range(271000, 271016), heldout=True, slips=True)
        torch.save((train_rows, val_rows), CACHE)
    # Only RGB observations inside an unresolved grasp attempt teach recovery.
    train = pack([row for row in train_rows if row[6] and row[3][-1] == 0], device)
    val = pack([row for row in val_rows if row[6] and row[3][-1] == 0], device)
    assert len(train[5]) > 20 and len(val[5]) > 5
    base = VisualActionHead().to(device).eval()
    base.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True)["state"])
    with torch.no_grad():
        train_logits = base(*train[:5])
        val_logits = base(*val[:5])
    head = RecoveryActionHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -1
    for step in range(801):
        if step:
            ids = torch.randint(len(train[5]), (32,), device=device)
            logits = head(train[3][ids], train_logits[ids])
            loss = torch.nn.functional.cross_entropy(logits, train[5][ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % 50 == 0:
            head.eval()
            with torch.no_grad():
                predictions = head(val[3], val_logits).argmax(-1)
            correct = int((predictions == val[5]).sum())
            # The first response to failure must be learned too, not just the continuation.
            first = val[5] == SKILLS.index("open")
            first_correct = int(((predictions == val[5]) & first).sum())
            rank = correct + 2 * first_correct
            print(f"step={step} recovery={correct}/{len(val[5])} first={first_correct}/{int(first.sum())}", flush=True)
            if rank > best:
                best = rank
                torch.save({"state": {k: v.cpu() for k, v in head.state_dict().items()},
                            "validation": {"correct": correct, "total": len(val[5]),
                                           "first_correct": first_correct, "first_total": int(first.sum())},
                            "step": step}, RECOVERY_CHECKPOINT)
            head.train()
    RECOVERY_CHECKPOINT.with_suffix(".json").write_text(json.dumps({
        "train": len(train[5]), "validation": len(val[5]), "best_rank": best,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
