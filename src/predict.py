"""Final predictions for the 1260 submission users.

Train on ISO weeks 11-30, pool = tv_show_id != 0 from the 31-42 schedule,
Hybrid (RRF History + ALS + Popularity) with the best config from results.md.

Usage: python src/predict.py   (writes submission_hybrid.csv, submission_popularity.csv)
"""
import numpy as np
import pandas as pd

from eval import (DATA, ROOT, SEED, als, build_matrix, finalize, fuse, history, history_days, popularity,
                  popularity_days)

TRAIN_WEEKS = (11, 30)
POP_WEEKS = (29, 30)
POOL_FILE = "export_arh_31-42-final.csv"
K0, W_HIST, W_ALS, W_POP = 10, 1, 0.5, 0.5
ALS_FACTORS, ALS_REG = 32, 0.1


def check(sub: pd.DataFrame, template: pd.DataFrame, pool_set: set) -> None:
    assert len(sub) == 1260 == len(template), len(sub)
    assert (sub.user_id.to_numpy() == template.user_id.to_numpy()).all(), "user order differs from template"
    assert list(sub.columns) == list(template.columns), sub.columns
    for s in sub.tv_show_id:
        ids = [int(x) for x in s.split(" ")]
        assert len(ids) == 5 and len(set(ids)) == 5, s
        assert 0 not in ids, s
        assert all(i in pool_set for i in ids), s


def main():
    inter = pd.read_parquet(DATA / "interactions.parquet")
    train = inter[inter.iso_week.between(*TRAIN_WEEKS)]
    train_us = train.groupby(["user_id", "tv_show_id"], as_index=False).overlap_sec.sum()

    pool_ids = pd.read_csv(DATA / POOL_FILE, usecols=["tv_show_id"]).tv_show_id
    pool = np.sort(pool_ids[pool_ids != 0].unique())
    pool_set = set(pool.tolist())

    template = pd.read_csv(DATA / "submission.csv", dtype=str)
    users = template.user_id.astype("uint64").tolist()
    train_users = set(train_us.user_id)

    pop = popularity(train, pool, weeks=POP_WEEKS)
    hist = history(train_us, pool, set(users))
    mat, _, iids, u_idx, _ = build_matrix(train_us)
    als_recs = als(mat, iids, u_idx, pool, users, ALS_FACTORS, ALS_REG)

    print(f"train weeks {TRAIN_WEEKS}: {len(train)} interactions, {len(train_us)} user-show pairs, {len(train_users)} users")
    print(f"pool ({POOL_FILE}): {len(pool)} shows, in train items: {len(pool_set & set(train_us.tv_show_id))}")
    print(f"popularity list (weeks {POP_WEEKS}): {len(pop)} items, top-5 {pop[:5]}")
    print(f"submission users: {len(users)}; cold (no train): {sum(u not in train_users for u in users)}; "
          f"with pool history: {sum(u in hist for u in users)}; with ALS recs: {sum(u in als_recs for u in users)}")

    hist_d = history_days(train, pool, set(users))
    pop_d = popularity_days(train, pool, POP_WEEKS)
    print(f"popularity by user-days (weeks {POP_WEEKS}): top-5 {pop_d[:5]}")

    def hybrid(h):
        return [finalize(fuse([h.get(u, []), als_recs.get(u, []), pop], (W_HIST, W_ALS, W_POP), K0), pop)
                for u in users]

    variants = {
        "submission_hybrid.csv": hybrid(hist),
        "submission_popularity.csv": [finalize([], pop)] * len(users),
        "submission_history_days.csv": [finalize(hist_d.get(u, []), pop) for u in users],
        "submission_hybrid_days.csv": hybrid(hist_d),
        "submission_popularity_days.csv": [finalize([], pop_d)] * len(users),
    }
    outputs = {
        name: pd.DataFrame({"user_id": template.user_id, "tv_show_id": [" ".join(map(str, r)) for r in recs]})
        for name, recs in variants.items()
    }
    for name, sub in outputs.items():
        check(sub, template, pool_set)
        sub.to_csv(ROOT / name, index=False)
        print(f"{name}: OK (1260 rows, 5 distinct non-zero pool ids each, template order); "
              f"unique lists: {sub.tv_show_id.nunique()}; first row: {sub.iloc[0].tolist()}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
