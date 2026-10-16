# -*- coding: utf-8 -*-
"""
Chronos-2 파이프라인의 공통 코드

  ※ 기존 kNN 작업(model_training 폴더)은 '읽기만' 하고 절대 수정하지 않는다.
     - 읽는 파일: model_training/data/market_state_features.csv, train_table.csv,
                  model_training/results/step6_test_predictions.csv (kNN 결과 비교용)
     - 쓰는 곳  : chronos2_training/results, chronos2_training/models (이 폴더 안)

설계 (기획서의 정의를 그대로 따른다)
  기준시각 = 한국 거래일 오후 6시. 한 날짜(t)에서 "t 이후 5거래일"을 예측한다.
  예측 대상(타깃)   : 코스피 '후행 5일 수익률'(%) 시계열 = kospi_ret_5d_pct
                      → 예측 5번째 단계(h=5)의 값이 곧 train_table 의 정답
                        target_kospi_ret_fwd5d_pct (t 종가 대비 5거래일 뒤 수익률)
  미래에 알려진 공변량 : 미국 10년 금리 후행 5일 변화(bp), 원달러 후행 5일 변화(%)
                      → h=5 단계의 값이 곧 가정 충격 shock_us10y_fwd5d_bp / shock_usdkrw_fwd5d_pct
  과거 공변량        : VIX, 코스피 20일 변동성, 코스피 20일 수익률 (kNN 핵심 입력과 같은 정보)
  (위 세 가지 항등식은 selfcheck.py 가 실제 데이터로 확인한다.)

Chronos-2 는 날짜 간격이 규칙적인(빈 날짜가 없는) 시계열만 받는다. 거래일에는 주말·휴일이 빠져 있으므로
'거래일 번호'를 하루 간격 가상 날짜로 바꿔서 넣는다(모델은 값만 보고 날짜는 간격 검사에만 쓰인다).
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

# ── 경로 ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]          # chronos2_training 폴더
PROJECT = ROOT.parent                               # 3차 프로젝트 폴더
KNN = PROJECT / "model_training"                    # 기존 kNN 작업 (읽기 전용)
STATE_CSV = KNN / "data" / "market_state_features.csv"
TRAIN_CSV = KNN / "data" / "train_table.csv"
KNN_TEST_PRED = KNN / "results" / "step6_test_predictions.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

# ── 컬럼·상수 ────────────────────────────────────────────
H = 5                                               # 예측 거래일 수
TAUS = (0.1, 0.5, 0.9)                              # 하단 10%, 중앙, 상단 90%
TARGET = "kospi_ret_5d_pct"                         # 타깃: 코스피 후행 5일 수익률(%)
KNOWN = ["us_treasury_10y_chg_5d_bp", "usdkrw_ret_5d_pct"]   # 미래에 알려진 공변량(=충격)
PAST = ["vix", "kospi_vol_20d_ann_pct", "kospi_ret_20d_pct"]  # 과거 공변량 (kNN 핵심 입력과 같은 정보)
RATE_LVL, FX_LVL = "us_treasury_10y", "usdkrw"      # 충격 경로를 만들 때 쓰는 수준값
SHOCK_COLS = ["shock_us10y_fwd5d_bp", "shock_usdkrw_fwd5d_pct"]   # train_table 의 충격 2개
Y_RET = "target_kospi_ret_fwd5d_pct"                # train_table 의 정답(5일 뒤 수익률)

# 학습표 구조정리.xlsx 의 '시장 상태 입력 18개' 중, 타깃(kospi_ret_5d)과 충격으로 이미 쓰는 3개를 뺀 나머지 15개
ALL_PAST = ["kospi_ret_20d_pct", "kospi_vol_20d_ann_pct", "kosdaq_ret_5d_pct",
            "usdkrw", "vix", "vix_chg_5d_pt",
            "us_treasury_10y", "us_curve_slope_10y2y_pp", "bok_base_rate", "kr_treasury_3y",
            "kospi_foreign_flow_5d_z", "kospi_institution_flow_5d_z",
            "kr_cpi_yoy_pct", "us_cpi_yoy_pct", "us_unemployment_change_1m_pp"]
# 직접 고른 구성: 변동성 상태(VIX·코스피 변동성·20일 수익률) + 국내 수급·주변 시장(코스닥·외국인 수급).
# 학습 기간(2001~2018)만으로 '충격 2개로 설명하고 남은 변동 폭'과 상관이 있는 열만 추렸다 (근거는 README).
SELECT_PAST = ["vix", "kospi_vol_20d_ann_pct", "kospi_ret_20d_pct", "kosdaq_ret_5d_pct", "kospi_foreign_flow_5d_z"]

# 입력 구성 (제로샷에서 어떤 정보가 도움이 되는지 비교하려고 나눠 둠). 값 = (과거 공변량, 미래에 알려진 공변량)
VARIANTS = {
    "all": (ALL_PAST, KNOWN),       # 구성 1: 기존 입력 20개를 모두 사용 (과거 15 + 충격 2 + 타깃 자신)
    "select": (SELECT_PAST, KNOWN), # 구성 2: 직접 고른 5개 + 충격 2
    "full": (PAST, KNOWN),          # 처음 구성: 과거 공변량 3 + 충격 2
    "shock_only": ([], KNOWN),      # 충격 2 만
    "no_cov": ([], []),             # 코스피 시계열만 (충격을 넣어도 반응하지 않는 기준)
    # ── 충격 효과를 떼어 보는 비교용 (full 과 과거 공변량 3개는 같고 충격 열만 다름. 열 자체를 제거하며 0 으로 채우지 않음)
    "past_only": (PAST, []),                 # B: 코스피 + 과거 3 (충격 없음)  ← E(full) 와 비교하면 충격 2개의 순수 기여
    "rate_only": (PAST, [KNOWN[0]]),         # C: B + 금리 충격
    "fx_only": (PAST, [KNOWN[1]]),           # D: B + 환율 충격
}

# 충격이 5일에 걸쳐 어떻게 들어오는지(누적 비율). 기본은 일정하게 나눠서.
SHAPES = {
    "linear": np.array([.2, .4, .6, .8, 1.0]),
    "front": np.ones(5),                            # 첫날 한 번에
    "back": np.array([0., 0., 0., 0., 1.0]),        # 마지막 날 한 번에
}

TS0 = pd.Timestamp("2000-01-03")                    # 가상 날짜의 시작 (거래일 번호 0)


# ── 데이터 ──────────────────────────────────────────────
def load_state() -> pd.DataFrame:
    """연속 일별 시장 상태표. 모든 입력 열이 채워진 첫 행(2001-03-02)부터 쓴다.
    월 단위 거시지표(미국 물가·실업률)의 맨 끝쪽 빈칸(2025-12~2026-01)은 '마지막 확정값'으로 이어 붙인다.
    (설계서의 '최근 확정값 자동 입력' 규칙과 같고, 앞의 값만 쓰므로 미래 정보가 들어가지 않는다.)"""
    s = pd.read_csv(STATE_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    need = [TARGET] + ALL_PAST + KNOWN
    raw_na = s[need].isna()
    s[need] = s[need].ffill()
    first = int(s[need].notna().all(axis=1).idxmax())
    n_filled = int(raw_na.iloc[first:].sum().sum())
    s = s.iloc[first:].reset_index(drop=True)
    assert s[need].notna().all().all(), "필요한 열에 빈칸이 있습니다"
    s.attrs["ffilled_cells"] = n_filled
    return s


def load_truth() -> pd.DataFrame:
    """기존 학습표(정답·분할). 평가는 kNN 과 똑같은 행·정답으로 한다."""
    t = pd.read_csv(TRAIN_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return t


class Market:
    """연속 일별 표를 numpy 로 들고 있으면서 컨텍스트·미래 공변량 표를 만들어 준다."""

    def __init__(self, state: pd.DataFrame):
        self.state = state
        self.n = len(state)
        self.dates = pd.to_datetime(state["date"]).to_numpy()
        self.rate = state[RATE_LVL].to_numpy(float)
        self.fx = state[FX_LVL].to_numpy(float)
        self.col = {c: state[c].to_numpy(float) for c in dict.fromkeys([TARGET] + ALL_PAST + KNOWN)}
        self._pos = {pd.Timestamp(d): i for i, d in enumerate(self.dates)}

    def pos(self, date) -> int:
        return self._pos[pd.Timestamp(date)]

    def date(self, i: int) -> pd.Timestamp:
        return pd.Timestamp(self.dates[i])

    @staticmethod
    def ts(pos):
        """거래일 번호 → 하루 간격 가상 날짜."""
        return TS0 + pd.to_timedelta(np.asarray(pos), unit="D")

    def context(self, i: int, L: int, cols: list[str], key: str) -> pd.DataFrame:
        """기준일 i(포함)까지 최근 L거래일. 타깃과 공변량."""
        lo = max(0, i - L + 1)
        d = {"id": key, "timestamp": self.ts(np.arange(lo, i + 1)), TARGET: self.col[TARGET][lo:i + 1]}
        for c in cols:
            d[c] = self.col[c][lo:i + 1]
        return pd.DataFrame(d)

    def future_realized(self, i: int, key: str) -> pd.DataFrame:
        """실제로 일어난 i+1 ~ i+5 의 후행 5일 변화 (과거 검증용)."""
        assert i + H < self.n
        p = np.arange(i + 1, i + 1 + H)
        d = {"id": key, "timestamp": self.ts(p)}
        for c in KNOWN:
            d[c] = self.col[c][i + 1:i + 1 + H]
        return pd.DataFrame(d)

    def future_ramp(self, i: int, key: str, rate_bp: float, fx_pct: float, shape: str = "linear") -> pd.DataFrame:
        """5일 누적 충격(금리 rate_bp, 환율 fx_pct)을 일별 경로로 풀어 '후행 5일 변화'를 만든다.
        h=5 번째 값은 정확히 rate_bp, fx_pct 가 된다 (train_table 의 충격 정의와 동일).
        h<5 는 5일 창에 과거 값이 섞이므로 기준일 이전 수준값을 함께 쓴다."""
        w = SHAPES[shape]
        r_hist, f_hist = self.rate[i - 4:i + 1], self.fx[i - 4:i + 1]       # 위치 i-4 .. i
        r_path = np.concatenate([r_hist, self.rate[i] + rate_bp / 100.0 * w])  # 금리(%) 수준 경로
        f_path = np.concatenate([f_hist, self.fx[i] * (1.0 + fx_pct / 100.0 * w)])
        h = np.arange(1, H + 1)
        rate_chg = (r_path[4 + h] - r_path[h - 1]) * 100.0                  # bp
        fx_chg = (f_path[4 + h] / f_path[h - 1] - 1.0) * 100.0              # %
        return pd.DataFrame({"id": key, "timestamp": self.ts(np.arange(i + 1, i + 1 + H)),
                             KNOWN[0]: rate_chg, KNOWN[1]: fx_chg})


# ── 요청 만들기 · 예측 ──────────────────────────────────────
def build_requests(mk: Market, specs: list[dict], L: int, variant: str, shape: str = "linear"):
    """specs: [{id, origin(거래일 번호), rate_bp, fx_pct, realized(선택)}] → (context_df, future_df)"""
    past_cols, known_cols = VARIANTS[variant]
    cols = list(past_cols) + list(known_cols)
    ctx, fut = [], []
    for sp in specs:
        i, key = sp["origin"], sp["id"]
        ctx.append(mk.context(i, L, cols, key))
        if known_cols:
            f = mk.future_realized(i, key) if sp.get("realized") else \
                mk.future_ramp(i, key, sp["rate_bp"], sp["fx_pct"], shape)
            fut.append(f[["id", "timestamp"] + list(known_cols)])
    ctx_df = pd.concat(ctx, ignore_index=True)
    fut_df = pd.concat(fut, ignore_index=True) if fut else None
    return ctx_df, fut_df


def forecast(pipe, mk: Market, specs: list[dict], L: int, variant: str, shape: str = "linear",
             batch_items: int = 64, quantiles=TAUS, log=print) -> pd.DataFrame:
    """Chronos-2 로 H=5 단계 예측. 돌려주는 표: id, step(1~5), timestamp, 분위수 열("0.1" 등), predictions"""
    out, qcols = [], [str(q) for q in quantiles]
    for k in range(0, len(specs), batch_items):
        part = specs[k:k + batch_items]
        ctx_df, fut_df = build_requests(mk, part, L, variant, shape)
        kw = dict(id_column="id", timestamp_column="timestamp", target=TARGET,
                  prediction_length=H, quantile_levels=list(quantiles))
        if fut_df is not None:
            kw["future_df"] = fut_df
        pred = pipe.predict_df(ctx_df, **kw).sort_values(["id", "timestamp"])
        pred["step"] = pred.groupby("id").cumcount() + 1
        out.append(pred[["id", "step", "timestamp"] + qcols + ["predictions"]])
        if log and (k // batch_items) % 5 == 0:
            log(f"    예측 {min(k + batch_items, len(specs)):,}/{len(specs):,}")
    return pd.concat(out, ignore_index=True)


class FakePipeline:
    """chronos 없이 입력·출력 '배관'만 점검하는 모의 예측기. 실제 모델이 아니며 성능 비교에 쓰면 안 된다."""

    def predict_df(self, df, future_df=None, id_column="item_id", timestamp_column="timestamp",
                   target="target", prediction_length=H, quantile_levels=TAUS, **kw):
        rows = []
        nd = NormalDist()
        for key, g in df.groupby(id_column, sort=False):
            last = g[timestamp_column].iloc[-1]
            sd = float(g[target].tail(250).std())
            f = None if future_df is None else future_df[future_df[id_column] == key].reset_index(drop=True)
            for h in range(1, prediction_length + 1):
                shift = 0.0 if (f is None or KNOWN[1] not in f.columns) \
                    else -0.4 * float(f[KNOWN[1]].iloc[h - 1]) * h / prediction_length
                r = {id_column: key, timestamp_column: last + pd.Timedelta(days=h),
                     "target_name": target, "predictions": shift}
                for q in quantile_levels:
                    r[str(q)] = shift + nd.inv_cdf(q) * sd * np.sqrt(h / prediction_length)
                rows.append(r)
        return pd.DataFrame(rows)


def load_pipeline(model: str, device: str = "auto"):
    """Chronos-2 를 불러온다. model: 'amazon/chronos-2' 같은 허브 이름 또는 파인튜닝 체크포인트 폴더."""
    import torch
    from chronos import Chronos2Pipeline
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"모델 불러오는 중: {model} (device={device})")
    return Chronos2Pipeline.from_pretrained(model, device_map=device)


def read_meta(model: str):
    """파인튜닝 체크포인트의 학습 기록(train_meta.json). 허브 모델(제로샷)이면 None."""
    p = Path(model) / "train_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# ── 평가 지표 (model_training/src/common.py 와 같은 정의) ──────────
def pinball(y, q, tau):
    e = np.asarray(y) - np.asarray(q)
    return np.maximum(tau * e, (tau - 1) * e)


def range_metrics(y, q10, q50, q90):
    y, q10, q50, q90 = map(np.asarray, (y, q10, q50, q90))
    loss_rows = (pinball(y, q10, .1) + pinball(y, q50, .5) + pinball(y, q90, .9)) / 3
    return {
        "pinball": loss_rows.mean(),
        "coverage80": np.mean((y >= q10) & (y <= q90)),
        "width80": np.mean(q90 - q10),
        "mae_median": np.mean(np.abs(y - q50)),
        "crossing": np.mean((q10 > q50) | (q50 > q90)),
    }, loss_rows


def block_bootstrap_skill(loss_model, loss_base, block=20, B=2000, seed=0):
    """skill = 1 - 모델 손실/기준 손실 의 95% 구간 (20거래일 묶음 복원추출)."""
    rng = np.random.default_rng(seed)
    n = len(loss_model)
    starts = np.arange(0, n, block)
    out = []
    for _ in range(B):
        pick = rng.choice(starts, len(starts))
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in pick])
        out.append(1 - loss_model[idx].mean() / loss_base[idx].mean())
    return np.percentile(out, [2.5, 97.5])


def block_bootstrap_diff(loss_a, loss_b, block=20, B=2000, seed=0):
    """두 모델의 날짜별 손실 차이(평균 a − b)의 95% 구간과 skill(1 − a/b)의 95% 구간.
    같은 날짜의 a·b 를 함께 뽑는 짝지은(paired) 20거래일 블록 부트스트랩. 구간이 0 을 포함하면 차이를 확신할 수 없다."""
    rng = np.random.default_rng(seed)
    n = len(loss_a)
    starts = np.arange(0, n, block)
    d, sk = [], []
    for _ in range(B):
        pick = rng.choice(starts, len(starts))
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in pick])
        d.append(loss_a[idx].mean() - loss_b[idx].mean())
        sk.append(1 - loss_a[idx].mean() / loss_b[idx].mean())
    return np.percentile(d, [2.5, 97.5]), np.percentile(sk, [2.5, 97.5])


def sigma_5d(df):
    """변동성 스케일 기준선의 예상 변동 폭(5거래일, %p)."""
    return np.sqrt(0.5 * df["kospi_vol_20d_ann_pct"] ** 2 + 0.5 * df["vix"] ** 2) * np.sqrt(5 / 252)


def baselines(dev: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """기준선 ① 과거 분포 ② 변동성 스케일 (dev 로 학습, rows 예측). kNN 6단계와 같은 방식."""
    q1 = dev[Y_RET].quantile(TAUS).to_numpy()
    zq = (dev[Y_RET] / sigma_5d(dev)).quantile(TAUS).to_numpy()
    return {"M1_unconditional": np.tile(q1, (len(rows), 1)),
            "M2_volscale": sigma_5d(rows).to_numpy()[:, None] * zq[None, :]}


def score(models: dict, y, vix, vix_hi) -> pd.DataFrame:
    """models: {이름: (행수, 3) 하단·중앙·상단 예측}. M1_unconditional, M2_volscale 이 반드시 들어 있어야 한다."""
    y, vix = np.asarray(y), np.asarray(vix)
    hi = vix > vix_hi
    rows, losses = [], {}
    for name, q in models.items():
        q = np.asarray(q, float)
        cross = float(np.mean((q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2])))
        qs = np.sort(q, axis=1)                              # 분위수가 엇갈리면 정렬해서 채점 (엇갈림 비율은 따로 기록)
        m, loss = range_metrics(y, qs[:, 0], qs[:, 1], qs[:, 2])
        m["crossing"] = cross
        ins = (y >= qs[:, 0]) & (y <= qs[:, 2])
        m.update(model=name, n=len(y), below10=float(np.mean(y < qs[:, 0])), above10=float(np.mean(y > qs[:, 2])),
                 coverage80_vix_low=float(ins[~hi].mean()), coverage80_vix_high=float(ins[hi].mean()))
        losses[name] = loss
        rows.append(m)
    b1, b2 = losses["M1_unconditional"], losses["M2_volscale"]
    for r in rows:
        l = losses[r["model"]]
        r["skill_vs_M1"] = 1 - l.mean() / b1.mean()
        r["skill_vs_M2"] = 1 - l.mean() / b2.mean()
        if r["model"] not in ("M1_unconditional", "M2_volscale"):
            lo, up = block_bootstrap_skill(l, b1)
            r["ci_vs_M1"] = f"{lo:+.3f}~{up:+.3f}"
            lo, up = block_bootstrap_skill(l, b2)
            r["ci_vs_M2"] = f"{lo:+.3f}~{up:+.3f}"
    cols = ["model", "n", "pinball", "skill_vs_M1", "ci_vs_M1", "skill_vs_M2", "ci_vs_M2", "coverage80",
            "below10", "above10", "coverage80_vix_low", "coverage80_vix_high", "width80", "mae_median", "crossing"]
    return pd.DataFrame(rows).reindex(columns=cols)


# ── 구간 보정: 하단·상단이 각각 10% 이탈하도록 범위를 늘이거나 줄이는 배율 ───────
CAL_CLIP = (0.25, 4.0)                                # 배율의 하한·상한 (이상한 값 방지)


def _tail_ratios(y, q):
    """중앙값 기준으로, 정답이 예측 하단·상단 거리의 몇 배 지점에 있는지 (하단 쪽, 상단 쪽)."""
    q = np.sort(np.asarray(q, float), axis=1)
    d_lo = np.maximum(q[:, 1] - q[:, 0], 1e-6)
    d_hi = np.maximum(q[:, 2] - q[:, 1], 1e-6)
    y = np.asarray(y, float)
    return (q[:, 1] - y) / d_lo, (y - q[:, 1]) / d_hi


def _k(r, alpha):
    return float(np.clip(np.quantile(r, 1 - alpha), *CAL_CLIP))


def fit_calibration(y, q, vix, vix_hi, alpha=0.10, min_group=60) -> dict:
    """보정 기간의 정답 y 와 예측 q(행수, 3)로, 하단·상단 이탈이 각각 alpha(10%)가 되는 배율을 구한다.
    중앙값에서 하단·상단까지의 거리에 곱하는 배율이라, 예측 범위가 넓은 날은 더 넓게 보정된다.
    global : 전체에 한 쌍 / by_vix : VIX 가 낮은 날·높은 날 각각 한 쌍 (표본이 min_group 미만이면 global 로 대신)"""
    y, vix = np.asarray(y, float), np.asarray(vix, float)
    r_lo, r_hi = _tail_ratios(y, q)
    cal = {"alpha": alpha, "n": int(len(y)), "vix_hi": float(vix_hi),
           "global": {"k_lo": _k(r_lo, alpha), "k_hi": _k(r_hi, alpha)}}
    hi = vix > vix_hi
    by = {}
    for name, m in (("low", ~hi), ("high", hi)):
        if m.sum() >= min_group:
            by[name] = {"k_lo": _k(r_lo[m], alpha), "k_hi": _k(r_hi[m], alpha), "n": int(m.sum())}
        else:
            by[name] = {**cal["global"], "n": int(m.sum())}
    cal["by_vix"] = by
    return cal


def apply_calibration(q, vix, cal: dict, mode: str = "global") -> np.ndarray:
    """보정 배율을 적용한 (행수, 3) 예측. 중앙값은 그대로 두고 하단·상단만 늘이거나 줄인다.
    mode: 'global'(하단·상단 각각 전체 배율) / 'by_vix'(VIX 구간별) / 'lower'(하단만 늘이고 상단은 그대로 — 손실 쪽 범위만 보정)."""
    q = np.sort(np.asarray(q, float), axis=1)
    d_lo, d_hi = q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]
    if mode == "lower":
        return np.column_stack([q[:, 1] - cal["global"]["k_lo"] * d_lo, q[:, 1], q[:, 2]])
    if mode == "by_vix":
        hi = np.asarray(vix, float) > cal["vix_hi"]
        k_lo = np.where(hi, cal["by_vix"]["high"]["k_lo"], cal["by_vix"]["low"]["k_lo"])
        k_hi = np.where(hi, cal["by_vix"]["high"]["k_hi"], cal["by_vix"]["low"]["k_hi"])
    else:
        k_lo, k_hi = cal["global"]["k_lo"], cal["global"]["k_hi"]
    return np.column_stack([q[:, 1] - k_lo * d_lo, q[:, 1], q[:, 1] + k_hi * d_hi])


def load_calibration(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
