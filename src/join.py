"""Join the viewing log with TV schedules into user-show interactions.

Usage: python src/join.py
Writes data/interactions.parquet and prints diagnostics.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
SCHEDULE_FILES = [
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
]
LOG_FILE = "dataset11-30.csv"
MIN_OVERLAP = 300
# Channel and time are packed into one int64 key so that a single global
# searchsorted works across all channels; unix seconds stay below 1e10.
KEY_MULT = 10**10
SEED = 42


def to_unix(ts: pd.Series) -> np.ndarray:
    return ts.astype("datetime64[s]").astype("int64").to_numpy()


def load_schedule() -> pd.DataFrame:
    parts = []
    for f in SCHEDULE_FILES:
        s = pd.read_csv(
            DATA / f,
            usecols=["channel_id", "start_time", "duration", "tv_show_id", "tv_show_title"],
            dtype={"channel_id": "int64", "duration": "int64", "tv_show_id": "int64"},
        )
        s["src_file"] = f
        s["src_row"] = np.arange(len(s))  # 0-based data row, header excluded
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    s["start_time"] = pd.to_datetime(s.start_time, format="%d.%m.%Y %H:%M:%S")
    s["end_time"] = s.start_time + pd.to_timedelta(s.duration, unit="s")

    n = len(s)
    s = s.drop_duplicates(["channel_id", "start_time"], keep="first")
    print(f"schedule: {n} rows, {n - len(s)} duplicates by (channel_id, start_time) dropped, {len(s)} left")
    s = s.sort_values(["channel_id", "start_time"], ignore_index=True)

    prev_end = s.groupby("channel_id").end_time.shift()
    print(f"schedule: rows starting before previous program on channel ends: {(s.start_time < prev_end).sum()}")
    print(f"schedule: zero/negative duration rows: {(s.duration <= 0).sum()}")
    return s


def load_log(channels: set) -> pd.DataFrame:
    log = pd.read_csv(
        DATA / LOG_FILE,
        dtype={"user_id": "uint64", "vsetv_id": "int64", "duraton": "int64"},
    )
    log["log_row"] = np.arange(len(log))
    log["start_time"] = pd.to_datetime(log.start_time, format="%Y-%m-%d %H:%M:%S")
    log["stop_time"] = pd.to_datetime(log.stop_time, format="%Y-%m-%d %H:%M:%S")
    keep = log.vsetv_id.isin(channels)
    print(f"log: {len(log)} rows, dropped {(~keep).sum()} ({(~keep).mean():.4%}) with vsetv_id not in schedule")
    return log[keep].reset_index(drop=True)


def overlap_join(log: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """All (log row, program) pairs with positive overlap, no cartesian product.

    Programs are sorted by (channel, start). For a view [a, b) the candidates are
    programs from lo to hi, where lo is the first program whose running max of
    end_key is > a (programs before it all end <= a) and hi is the first program
    with start >= b. That handles overlapping programs; pairs are then filtered
    to overlap > 0.
    """
    p_start = sched.channel_id.to_numpy() * KEY_MULT + to_unix(sched.start_time)
    p_end = sched.channel_id.to_numpy() * KEY_MULT + to_unix(sched.end_time)
    p_end_cummax = np.maximum.accumulate(p_end)  # sorted by (channel, start), so also monotone

    v_start = log.vsetv_id.to_numpy() * KEY_MULT + to_unix(log.start_time)
    v_stop = log.vsetv_id.to_numpy() * KEY_MULT + to_unix(log.stop_time)

    lo = np.searchsorted(p_end_cummax, v_start, side="right")
    hi = np.searchsorted(p_start, v_stop, side="left")
    cnt = np.clip(hi - lo, 0, None)

    view_idx = np.repeat(np.arange(len(log)), cnt)
    offsets = np.repeat(np.cumsum(cnt) - cnt, cnt)
    prog_idx = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - offsets)

    ov = np.minimum(v_stop[view_idx], p_end[prog_idx]) - np.maximum(v_start[view_idx], p_start[prog_idx])
    pos = ov > 0
    print(f"join: {cnt.sum()} candidate pairs, {pos.sum()} with overlap > 0")
    return pd.DataFrame({"view_idx": view_idx[pos], "prog_idx": prog_idx[pos], "overlap_sec": ov[pos]})


def tz_check(log: pd.DataFrame, sched: pd.DataFrame, n: int = 200_000, tol: int = 60) -> None:
    sample = log.sample(n=min(n, len(log)), random_state=SEED)
    v = sample.vsetv_id.to_numpy() * KEY_MULT + to_unix(sample.start_time)
    ch = sched.channel_id.to_numpy() * KEY_MULT
    t = to_unix(sched.start_time)
    rows = []
    for h in range(-3, 4):
        keys = np.sort(ch + t + h * 3600)
        i = np.searchsorted(keys, v)
        right = np.abs(keys[np.clip(i, 0, len(keys) - 1)] - v)
        left = np.abs(keys[np.clip(i - 1, 0, len(keys) - 1)] - v)
        rows.append({"shift_h": h, "share_within_60s": (np.minimum(left, right) <= tol).mean()})
    res = pd.DataFrame(rows)
    print(f"\ntimezone check ({len(sample)} sampled views, any program, |view_start - prog_start| <= {tol}s):")
    print(res.to_string(index=False, float_format="%.4f"))
    best = int(res.loc[res.share_within_60s.idxmax(), "shift_h"])
    if best != 0:
        print(f"WARNING: maximum is at shift {best:+d}h, not 0. Schedule NOT corrected.")


def main() -> None:
    sched = load_schedule()
    labeled = sched.tv_show_id != 0
    print(f"schedule: labeled (tv_show_id != 0) {labeled.sum()}, unlabeled {(~labeled).sum()} kept separately for stats")

    log = load_log(set(sched.channel_id.unique()))
    tz_check(log, sched)

    pairs = overlap_join(log, sched)
    pairs["tv_show_id"] = sched.tv_show_id.to_numpy()[pairs.prog_idx]
    views = pairs[pairs.overlap_sec >= MIN_OVERLAP]

    # --- stats over all programs (labeled + unlabeled)
    print(f"\npairs with overlap >= {MIN_OVERLAP}s: {len(views)}; "
          f"share in labeled program: {(views.tv_show_id != 0).mean():.4f}")
    rows_any = views.view_idx.nunique()
    rows_lab = views.loc[views.tv_show_id != 0, "view_idx"].nunique()
    print(f"log rows with >=1 view (any program): {rows_any} ({rows_any / len(log):.4f} of kept log rows); "
          f"with >=1 labeled view: {rows_lab} ({rows_lab / len(log):.4f})")

    lv = views[views.tv_show_id != 0]
    inter = pd.DataFrame({
        "user_id": log.user_id.to_numpy()[lv.view_idx],
        "tv_show_id": lv.tv_show_id.to_numpy(),
        "channel_id": log.vsetv_id.to_numpy()[lv.view_idx],
        "view_start": log.start_time.to_numpy()[lv.view_idx],
        "overlap_sec": lv.overlap_sec.to_numpy(),
    })
    inter["iso_week"] = inter.view_start.dt.isocalendar().week.astype("int16")
    out = DATA / "interactions.parquet"
    inter.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(inter)} rows")

    def n_pairs(lo, hi):
        w = inter[inter.iso_week.between(lo, hi)]
        return len(w[["user_id", "tv_show_id"]].drop_duplicates())

    print(f"unique user-show pairs weeks 11-20: {n_pairs(11, 20)}; weeks 21-30: {n_pairs(21, 30)}")
    print(f"unique users: {inter.user_id.nunique()}, unique shows: {inter.tv_show_id.nunique()}")

    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": "uint64"})
    has = sub.user_id.isin(set(inter.user_id.unique()))
    print(f"test users with >=1 interaction: {has.sum()} of {len(sub)}")

    q = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    print("\noverlap_sec quantiles:")
    print(pd.DataFrame({
        "all overlap>0": pairs.overlap_sec.quantile(q),
        "interactions": inter.overlap_sec.quantile(q),
    }).to_string(float_format="%.0f"))

    # --- 3 random matches against raw source lines
    picks = lv.sample(3, random_state=SEED)
    need_log = set(log.log_row.to_numpy()[picks.view_idx])
    log_lines = {}
    with (DATA / LOG_FILE).open(encoding="utf-8") as fh:
        next(fh)
        for i, line in enumerate(fh):
            if i in need_log:
                log_lines[i] = line.rstrip("\n")
    for _, p in picks.iterrows():
        lr = log.iloc[int(p.view_idx)]
        sr = sched.iloc[int(p.prog_idx)]
        raw_s = pd.read_csv(DATA / sr.src_file, skiprows=range(1, int(sr.src_row) + 1), nrows=1, header=0)
        print("\n--- match")
        print(f"log line {int(lr.log_row)}: {log_lines[int(lr.log_row)]}")
        print(f"schedule {sr.src_file} row {int(sr.src_row)}: {raw_s.iloc[0].to_dict()}")
        print(f"program interval: {sr.start_time} -> {sr.end_time}; overlap_sec={int(p.overlap_sec)}")


if __name__ == "__main__":
    main()
