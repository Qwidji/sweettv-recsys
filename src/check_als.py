"""Sanity check: ALS top-5 for one user must consist of pool ids only.

Usage: python src/check_als.py
"""
import random

import numpy as np
import pandas as pd

from eval import DATA, POOL_FILE, SEED, als, apk, build_matrix, load


def main():
    _, train_us, truth, pool, test_users = load()
    mat, _, iids, u_idx, _ = build_matrix(train_us)
    users = sorted(u for u in truth if u in u_idx and u in test_users)
    user = random.Random(SEED).choice(users)

    recs = als(mat, iids, u_idx, pool, [user], factors=32, reg=0.1)[user][:5]
    pool_set = set(pool)
    titles = (pd.read_csv(DATA / POOL_FILE, usecols=["tv_show_id", "tv_show_title"])
              .drop_duplicates("tv_show_id").set_index("tv_show_id").tv_show_title)
    watched = set(train_us.loc[train_us.user_id == user, "tv_show_id"])

    print(f"user {user}: train shows={len(watched)}, valid truth={len(truth[user])}")
    for r, it in enumerate(recs, 1):
        print(f"  {r}. {it:>10}  in_pool={it in pool_set}  watched_in_train={it in watched}"
              f"  in_valid_truth={it in truth[user]}  {titles.get(it, '?')}")
    assert len(recs) == 5, len(recs)
    assert all(it in pool_set for it in recs), "ALS recommended an id outside the pool"
    print("OK: all 5 ids are in the pool")
    print(f"AP@5 (ALS factors=32, reg=0.1) = {apk(recs, truth[user]):.6f}")


if __name__ == "__main__":
    main()
