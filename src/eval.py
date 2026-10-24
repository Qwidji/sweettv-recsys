"""Offline evaluation of recommenders on data/interactions.parquet.

Protocol: train = ISO weeks 11-20, valid = 21-30. Candidate pool = unique
tv_show_id != 0 from the 21-30 schedule. Ground truth = shows a user watched in
valid. MAP@5 with denominator min(5, |truth|). Repeats allowed; cold users and
padding to 5 come from popularity.

Usage: python src/eval.py   (writes results.md)
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit warns about BLAS threading otherwise

import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCHEDULE_FILES = ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"]
POOL_FILE = "export_arh_21-30-final.csv"
TRAIN_WEEKS = (11, 20)
VALID_WEEKS = (21, 30)
K = 5
TOP_N = 50  # candidate list length per method, used by fusion
SEED = 42
# Scores reported by `kaggle competitions submissions` on 2026-10-24.
KAGGLE_SCORES = [
    ("submission_popularity.csv (unique viewers, weeks 29-30)", 0.27903, 0.27504),
    ("submission_hybrid.csv (RRF, history by overlap)", 0.32225, 0.29500),
    ("submission_history_days.csv (history by distinct days)", 0.35200, 0.32653),
    ("submission_hybrid_days.csv (RRF, history by distinct days)", 0.35425, 0.33745),
    ("submission_popularity_days.csv (user-days, weeks 29-30)", 0.29133, 0.28631),
]

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------- metric

def apk(recs, truth, k=K):
    if not truth:
        return 0.0
    hits, score = 0, 0.0
    for i, r in enumerate(recs[:k]):
        if r in truth:
            hits += 1
            score += hits / (i + 1)
    return score / min(k, len(truth))


def unit_test_map():
    cases = [
        # truth, recs, expected (hand-computed)
        ({1, 2, 3}, [1, 4, 2, 5, 6], (1 / 1 + 2 / 3) / 3),   # hits at 1 and 3
        ({7}, [3, 7, 8, 9, 10], (1 / 2) / 1),                 # hit at 2
        (set(range(10)), [0, 1, 2, 3, 4], 5 / 5),             # denominator min(5, 10) = 5
        ({1, 2}, [3, 4, 5, 6, 1], (1 / 5) / 2),               # hit at 5
        ({1, 2}, [9, 8, 7, 6, 5], 0.0),
    ]
    lines = []
    for truth, recs, exp in cases:
        got = apk(recs, truth)
        assert abs(got - exp) < 1e-12, (truth, recs, got, exp)
        lines.append(f"truth={sorted(truth)} recs={recs} AP@5={got:.4f} expected={exp:.4f} OK")
    mean = np.mean([c[2] for c in cases])
    lines.append(f"MAP@5 over the 5 cases = {mean:.4f}")
    return lines


# ---------------------------------------------------------------- data

def load():
    inter = pd.read_parquet(DATA / "interactions.parquet")
    train = inter[inter.iso_week.between(*TRAIN_WEEKS)]
    valid = inter[inter.iso_week.between(*VALID_WEEKS)]
    train_us = train.groupby(["user_id", "tv_show_id"], as_index=False).overlap_sec.sum()
    truth = valid.groupby("user_id").tv_show_id.agg(set).to_dict()

    pool_ids = pd.read_csv(DATA / POOL_FILE, usecols=["tv_show_id"]).tv_show_id
    pool = np.sort(pool_ids[pool_ids != 0].unique())
    test_users = set(pd.read_csv(DATA / "submission.csv", dtype={"user_id": "uint64"}).user_id)
    return train, train_us, truth, pool, test_users


def finalize(ranked, fallback, k=K):
    out, seen = [], set()
    for it in list(ranked) + list(fallback):
        if it not in seen:
            out.append(it)
            seen.add(it)
            if len(out) == k:
                break
    return out


# ---------------------------------------------------------------- methods

def popularity(train, pool, weeks=None):
    t = train if weeks is None else train[train.iso_week.between(*weeks)]
    cnt = t[t.tv_show_id.isin(pool)].groupby("tv_show_id").user_id.nunique()
    return cnt.sort_values(ascending=False, kind="stable").index.tolist()[:TOP_N]


def history(train_us, pool, users):
    t = train_us[train_us.tv_show_id.isin(pool) & train_us.user_id.isin(users)]
    t = t.sort_values(["user_id", "overlap_sec"], ascending=[True, False], kind="stable")
    return {u: g.tv_show_id.tolist()[:TOP_N] for u, g in t.groupby("user_id", sort=False)}


def popularity_days(train, pool, weeks):
    """Pool shows ranked by distinct (user, day) pairs in `weeks`; ties by unique viewers."""
    t = train[train.iso_week.between(*weeks) & train.tv_show_id.isin(pool)]
    t = t.assign(day=t.view_start.dt.normalize())
    g = t.groupby("tv_show_id").agg(viewers=("user_id", "nunique"))
    g["user_days"] = t.drop_duplicates(["tv_show_id", "user_id", "day"]).groupby("tv_show_id").size()
    return g.sort_values(["user_days", "viewers"], ascending=False, kind="stable").index.tolist()[:TOP_N]


def history_days(train, pool, users):
    """User's pool shows ranked by number of distinct days watched; ties by total overlap."""
    t = train[train.tv_show_id.isin(pool) & train.user_id.isin(users)]
    g = (t.assign(day=t.view_start.dt.normalize())
         .groupby(["user_id", "tv_show_id"], as_index=False)
         .agg(days=("day", "nunique"), overlap=("overlap_sec", "sum")))
    g = g.sort_values(["user_id", "days", "overlap"], ascending=[True, False, False], kind="stable")
    return {u: x.tv_show_id.tolist()[:TOP_N] for u, x in g.groupby("user_id", sort=False)}


def build_matrix(train_us):
    uids = np.sort(train_us.user_id.unique())
    iids = np.sort(train_us.tv_show_id.unique())
    u_idx = {u: i for i, u in enumerate(uids)}
    i_idx = {it: i for i, it in enumerate(iids)}
    rows = train_us.user_id.map(u_idx).to_numpy()
    cols = train_us.tv_show_id.map(i_idx).to_numpy()
    vals = np.log1p(train_us.overlap_sec.to_numpy() / 60).astype(np.float32)
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(uids), len(iids)))
    return m, uids, iids, u_idx, i_idx


def top_from_scores(scores, cand_items, users, n=TOP_N):
    n = min(n, scores.shape[1])
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    order = np.take_along_axis(part, np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable"), axis=1)
    return {u: cand_items[order[j]].tolist() for j, u in enumerate(users)}


def als(mat, iids, u_idx, pool, users, factors, reg, iterations=20):
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iterations,
                                    random_state=SEED, use_gpu=False)
    model.fit(mat, show_progress=False)
    in_pool = np.isin(iids, pool)
    cand = iids[in_pool]
    known = [u for u in users if u in u_idx]
    uf = model.user_factors[[u_idx[u] for u in known]]
    scores = uf @ model.item_factors[in_pool].T  # already-watched NOT filtered
    return top_from_scores(scores, cand, known)


def content(train_us, pool, users):
    cols = ["tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
            "director", "actors"]
    meta = pd.concat([pd.read_csv(DATA / f, usecols=cols, dtype=str) for f in SCHEDULE_FILES])
    meta["tv_show_id"] = meta.tv_show_id.astype("int64")
    meta = meta[meta.tv_show_id != 0].drop_duplicates("tv_show_id").set_index("tv_show_id")

    def tokens(r):
        t = []
        if isinstance(r.tv_show_category, str):
            t.append("cat=" + r.tv_show_category)
        for g in (r.tv_show_genre_1, r.tv_show_genre_2, r.tv_show_genre_3):
            if isinstance(g, str):
                t.append("genre=" + g.strip())
        if isinstance(r.director, str):
            t += ["dir=" + d.strip() for d in r.director.split(",") if d.strip()]
        if isinstance(r.actors, str):
            t += ["act=" + a.strip() for a in r.actors.split(",") if a.strip()]
        return t

    docs = [tokens(r) for r in meta.itertuples()]
    vec = TfidfVectorizer(analyzer=lambda x: x)
    X = normalize(vec.fit_transform(docs))
    s_idx = {s: i for i, s in enumerate(meta.index)}

    t = train_us[train_us.user_id.isin(users) & train_us.tv_show_id.isin(s_idx)]
    uids = np.sort(t.user_id.unique())
    u_pos = {u: i for i, u in enumerate(uids)}
    W = sp.csr_matrix((np.log1p(t.overlap_sec.to_numpy() / 60),
                       (t.user_id.map(u_pos).to_numpy(), t.tv_show_id.map(s_idx).to_numpy())),
                      shape=(len(uids), X.shape[0]))
    W = sp.diags(1 / np.maximum(np.asarray(W.sum(1)).ravel(), 1e-12)) @ W  # weighted mean
    P = normalize(W @ X)
    cand = np.array([p for p in pool if p in s_idx])
    Xp = X[[s_idx[p] for p in cand]]
    scores = (P @ Xp.T).toarray().astype(np.float32)
    return top_from_scores(scores, cand, uids.tolist())


def fuse(lists, weights, k0):
    sc = {}
    for lst, w in zip(lists, weights):
        if w == 0:
            continue
        for r, it in enumerate(lst):
            sc[it] = sc.get(it, 0.0) + w / (k0 + r + 1)
    return sorted(sc, key=lambda x: -sc[x])


# ---------------------------------------------------------------- main

def hardware():
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except OSError:
            return "?"
    mem = sysctl("hw.memsize")
    mem = f"{int(mem) / 2**30:.0f} GB" if mem.isdigit() else mem
    return f"{sysctl('machdep.cpu.brand_string') or platform.processor()}, {sysctl('hw.ncpu')} CPU, {mem} RAM, {platform.platform()}"


def main():
    test_lines = unit_test_map()
    print("MAP@5 unit test:")
    print("\n".join("  " + l for l in test_lines))

    train, train_us, truth, pool, test_users = load()
    eval_users = sorted(truth)
    eval_test = [u for u in eval_users if u in test_users]
    train_users = set(train_us.user_id)
    pool_set = set(pool)
    truth_items = [it for s in truth.values() for it in s]
    facts = [
        f"train interactions: {len(train)}, user-show pairs: {len(train_us)}, users: {len(train_users)}",
        f"pool (tv_show_id != 0 in {POOL_FILE}): {len(pool)} shows; in train items: {len(pool_set & set(train_us.tv_show_id))}",
        f"eval users (non-empty valid truth): {len(eval_users)}; cold (no train): {sum(u not in train_users for u in eval_users)}",
        f"test users with non-empty valid truth: {len(eval_test)} of {len(test_users)}",
        f"truth user-show pairs: {len(truth_items)}; in pool: {np.mean([i in pool_set for i in truth_items]):.4f}",
    ]
    print("\n" + "\n".join(facts))

    def score(lists, fallback):
        aps = {u: apk(finalize(lists.get(u, []), fallback), truth[u]) for u in eval_users}
        return np.mean(list(aps.values())), np.mean([aps[u] for u in eval_test])

    rows = []

    def record(method, params, lists, fallback, secs):
        m_all, m_test = score(lists, fallback)
        rows.append({"method": method, "params": params, "map_all": m_all, "map_test": m_test, "sec": secs})
        print(f"{method:<12} {params:<40} MAP@5 all={m_all:.5f} test={m_test:.5f} ({secs:.1f}s)")

    print()
    t0 = time.perf_counter()
    pop = popularity(train, pool)
    record("Popularity", "unique viewers, weeks 11-20", {}, pop, time.perf_counter() - t0)

    t0 = time.perf_counter()
    pop2 = popularity(train, pool, weeks=(19, 20))
    record("Popularity", "unique viewers, weeks 19-20", {}, pop2, time.perf_counter() - t0)
    best_pop = pop if rows[0]["map_all"] >= rows[1]["map_all"] else pop2

    t0 = time.perf_counter()
    hist = history(train_us, pool, set(eval_users))
    record("History", "sum overlap desc, pad popularity", hist, best_pop, time.perf_counter() - t0)

    mat, _, iids, u_idx, _ = build_matrix(train_us)
    als_runs = {}
    for f in (32, 64):
        for reg in (0.01, 0.1):
            t0 = time.perf_counter()
            recs = als(mat, iids, u_idx, pool, eval_users, f, reg)
            secs = time.perf_counter() - t0
            als_runs[(f, reg)] = recs
            record("ALS", f"factors={f}, reg={reg}, iter=20", recs, best_pop, secs)
    als_rows = [r for r in rows if r["method"] == "ALS"]
    best_als_row = max(als_rows, key=lambda r: r["map_all"])
    best_als = als_runs[list(als_runs)[als_rows.index(best_als_row)]]

    t0 = time.perf_counter()
    cb = content(train_us, pool, set(eval_users))
    record("Content", "TF-IDF cat+genres+director+actors", cb, best_pop, time.perf_counter() - t0)

    t0 = time.perf_counter()
    grid = []
    # Weighted reciprocal-rank fusion; scale-invariant, so History weight is fixed to 1.
    for k0 in (10, 60):
        for w_als in (0.25, 0.5, 1, 2, 4):
            for w_pop in (0, 0.1, 0.25, 0.5):
                fused = {u: fuse([hist.get(u, []), best_als.get(u, []), best_pop], (1, w_als, w_pop), k0)
                         for u in eval_users}
                m_all, m_test = score(fused, best_pop)
                grid.append((m_all, m_test, k0, w_als, w_pop))
    grid.sort(reverse=True)
    m_all, m_test, k0, w_als, w_pop = grid[0]
    secs = time.perf_counter() - t0
    rows.append({"method": "Hybrid",
                 "params": f"RRF k0={k0}, w_hist=1, w_als={w_als}, w_pop={w_pop} (ALS {best_als_row['params'].split(', iter')[0]})",
                 "map_all": m_all, "map_test": m_test, "sec": secs})
    print(f"{'Hybrid':<12} {rows[-1]['params']:<40} MAP@5 all={m_all:.5f} test={m_test:.5f} ({secs:.1f}s, {len(grid)} configs)")
    print("hybrid top-5 configs (map_all, map_test, k0, w_als, w_pop):")
    for g in grid[:5]:
        print(f"  {g[0]:.5f} {g[1]:.5f} k0={g[2]} w_als={g[3]} w_pop={g[4]}")

    # --- "distinct days" variants
    print()
    t0 = time.perf_counter()
    hist_d = history_days(train, pool, set(eval_users))
    record("History-days", "distinct days desc, pad popularity", hist_d, best_pop, time.perf_counter() - t0)

    t0 = time.perf_counter()
    fused = {u: fuse([hist_d.get(u, []), best_als.get(u, []), best_pop], (1, w_als, w_pop), k0) for u in eval_users}
    record("Hybrid-days", f"RRF k0={k0}, w_hist=1, w_als={w_als}, w_pop={w_pop}, history by days",
           fused, best_pop, time.perf_counter() - t0)

    t0 = time.perf_counter()
    pop_d = popularity_days(train, pool, (19, 20))
    record("Popularity-days", "distinct (user, day) pairs, weeks 19-20", {}, pop_d, time.perf_counter() - t0)

    methods = {
        "Popularity": {},
        "History-days": hist_d,
        f"ALS ({best_als_row['params'].split(', iter')[0]})": best_als,
        "Content": cb,
        "Hybrid-days": fused,
    }
    show_lines = showcase(train, train_us, truth, eval_test, methods, best_pop)
    print("\n".join(show_lines))

    write_results(rows, facts, test_lines, grid, len(eval_users), len(eval_test), show_lines)


def showcase(train, train_us, truth, eval_test, methods, fallback):
    """Top-5 per method for the heaviest and lightest test user (by train interaction rows)."""
    n_inter = train.groupby("user_id").size()
    cand = n_inter[n_inter.index.isin(eval_test)].reset_index(name="n").sort_values(["n", "user_id"])
    light = cand.user_id.iloc[0]
    heavy = cand.sort_values(["n", "user_id"], ascending=[False, True]).user_id.iloc[0]

    meta = pd.concat([pd.read_csv(DATA / f, usecols=["tv_show_id", "tv_show_title", "tv_show_category"])
                      for f in [POOL_FILE] + [f for f in SCHEDULE_FILES if f != POOL_FILE]])
    meta = meta.drop_duplicates("tv_show_id").set_index("tv_show_id")

    lines = ["## Примеры: тяжёлый и лёгкий тестовый пользователь (valid)", "",
             "Выбраны среди тестовых пользователей с непустым valid truth и хотя бы одним train-взаимодействием: "
             "максимум и минимум числа строк взаимодействий в train (ничьи — по меньшему user_id). "
             "«train» — шоу уже смотрено пользователем в неделях 11–20.", ""]
    for label, u in (("Тяжёлый", heavy), ("Лёгкий", light)):
        watched = set(train_us.loc[train_us.user_id == u, "tv_show_id"])
        lines += [f"### {label}: user_id {u}", "",
                  f"Train: {int(n_inter[u])} взаимодействий, {len(watched)} разных шоу. "
                  f"Valid truth: {len(truth[u])} шоу.", "",
                  "| Метод | # | tv_show_id | Название | Категория | в truth | в train |",
                  "|---|---|---|---|---|---|---|"]
        summary = []
        for name, lists in methods.items():
            recs = finalize(lists.get(u, []), fallback)
            for r, it in enumerate(recs, 1):
                title = str(meta.tv_show_title.get(it, "?")).replace("|", "\\|")
                cat = meta.tv_show_category.get(it, "?")
                lines.append(f"| {name if r == 1 else ''} | {r} | {it} | {title} | {cat} | "
                             f"{'да' if it in truth[u] else '—'} | {'да' if it in watched else '—'} |")
            summary.append(f"| {name} | {apk(recs, truth[u]):.4f} | {sum(it in truth[u] for it in recs)} | "
                           f"{sum(it in watched for it in recs)} |")
        lines += ["", "| Метод | AP@5 | попаданий в truth | уже смотрел в train |", "|---|---|---|---|"]
        lines += summary + [""]
    return lines


def write_results(rows, facts, test_lines, grid, n_all, n_test, show_lines=()):
    vers = (f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "
            f"scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, implicit {implicit.__version__}")
    lines = [
        "# Results: offline validation", "",
        f"Протокол: train = ISO-недели {TRAIN_WEEKS[0]}–{TRAIN_WEEKS[1]}, valid = {VALID_WEEKS[0]}–{VALID_WEEKS[1]}. "
        f"Пул кандидатов: tv_show_id != 0 из `{POOL_FILE}`. MAP@5 со знаменателем min(5, |truth|). "
        "Повторы разрешены; холодные пользователи и добивка до 5 берутся из popularity "
        "(лучший вариант по valid). Seed = 42.", "",
        f"Пользователей в оценке: {n_all} (все с непустым truth), {n_test} из них тестовые.", "",
        "| Метод | Параметры | MAP@5 (все) | MAP@5 (тестовые) | Время, с |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['method']} | {r['params']} | {r['map_all']:.5f} | {r['map_test']:.5f} | {r['sec']:.1f} |")
    lines += ["", "Выбор лучшего ALS и весов Hybrid сделан по тому же valid, поэтому их оценки оптимистичны. "
              "Hybrid-days использует те же веса без повторного подбора.", "",
              "## Kaggle (обучение на неделях 11–30, пул 31-42, `src/predict.py`)", "",
              "| Сабмит | Public | Private |", "|---|---|---|"]
    lines += [f"| {name} | {pub} | {priv} |" for name, pub, priv in KAGGLE_SCORES]
    lines += ["",
              "## Hybrid: топ-5 конфигураций", "", "| MAP@5 (все) | MAP@5 (тестовые) | k0 | w_als | w_pop |",
              "|---|---|---|---|---|"]
    lines += [f"| {g[0]:.5f} | {g[1]:.5f} | {g[2]} | {g[3]} | {g[4]} |" for g in grid[:5]]
    lines += [""] + list(show_lines)
    lines += ["## Данные", ""] + [f"- {f}" for f in facts]
    lines += ["", "## Unit-тест MAP@5", "", "```"] + test_lines + ["```"]
    lines += ["", "## Окружение", "", f"- {vers}", f"- {hardware()}", ""]
    (ROOT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {ROOT / 'results.md'}")


if __name__ == "__main__":
    main()
