# -*- coding: utf-8 -*-
"""
Chronos-2 를 돌리기 전에 '데이터 준비가 맞는지' 확인하는 점검 (torch / chronos 없이도 실행됨)

  1) 후행 5일 변화 열이 수준값에서 다시 계산한 값과 같은가
  2) 예측 5번째 단계 = train_table 의 정답·충격과 같은가   (타깃·충격 정의가 기존 kNN 과 똑같다는 증거)
  3) 충격 경로(ramp)의 5번째 값이 입력한 충격과 정확히 같은가
  4) 만든 입력표의 날짜가 빈틈없이 하루 간격인가 (Chronos-2 의 필수 조건), chronos 가 설치돼 있으면 공식 검사 함수도 통과하는가
  5) 모의 예측기(FakePipeline)로 평가 흐름과 지표 계산이 끝까지 도는가
  6) 지표 계산이 기존 kNN 결과(model_card.json)와 같은 숫자를 내는가

실행: python src/selfcheck.py
"""
import json
import sys

import numpy as np
import pandas as pd

from common import (H, TAUS, TARGET, KNOWN, PAST, ALL_PAST, SELECT_PAST, VARIANTS, STATE_CSV, RATE_LVL, FX_LVL,
                    SHOCK_COLS, Y_RET, KNN, KNN_TEST_PRED, Market, load_state, load_truth, build_requests, forecast,
                    FakePipeline, baselines, score, fit_calibration, apply_calibration)
from finetune import make_items

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    s, t = load_state(), load_truth()
    mk = Market(s)
    print(f"일별 표: {mk.date(0):%Y-%m-%d} ~ {mk.date(mk.n - 1):%Y-%m-%d}  {mk.n:,}거래일 "
          f"(월 단위 지표 끝부분 {s.attrs.get('ffilled_cells', 0)}칸은 마지막 확정값으로 이어 붙임)\n")

    # 0) 입력 구성
    cols18 = [c for c in pd.read_csv(STATE_CSV, nrows=1).columns if c != "date"]
    check("기존 '시장 상태 입력 18개'를 빠짐없이 쓸 수 있음 (타깃 1 + 충격 2 + 과거 15)",
          len(cols18) == 18 and set(cols18) == {TARGET, *KNOWN, *ALL_PAST} and len(ALL_PAST) == 15,
          f"CSV 지표 열 {len(cols18)}개")
    check("직접 고른 구성(select)의 열이 모두 all 안에 있음", set(SELECT_PAST) <= set(ALL_PAST), f"{len(SELECT_PAST)}개")
    check("상태표에 빈칸이 없고 학습표와 같은 날(2001-03-02)부터 시작", s[[TARGET] + ALL_PAST + KNOWN].notna().all().all()
          and mk.date(0) == t["date"].min(), f"{mk.date(0):%Y-%m-%d}")

    # 1) 후행 5일 변화 재계산
    r, f = mk.rate, mk.fx
    rate5 = (r[5:] - r[:-5]) * 100.0
    fx5 = (f[5:] / f[:-5] - 1.0) * 100.0
    d_rate = np.abs(rate5 - mk.col[KNOWN[0]][5:]).max()
    d_fx = np.abs(fx5 - mk.col[KNOWN[1]][5:]).max()
    check("금리 후행 5일 변화(bp) = 수준값 차이 × 100", d_rate < 1e-6, f"최대 차이 {d_rate:.2e}")
    check("환율 후행 5일 변화(%) = 수준값 비율 − 1", d_fx < 1e-5, f"최대 차이 {d_fx:.2e}")

    # 2) 예측 5번째 단계 = train_table 정답·충격
    pos = np.array([mk.pos(d) for d in t["date"]])
    assert (pos + H < mk.n).all()
    d_y = np.abs(mk.col[TARGET][pos + H] - t[Y_RET].to_numpy()).max()
    d_r = np.abs(mk.col[KNOWN[0]][pos + H] - t[SHOCK_COLS[0]].to_numpy()).max()
    d_f = np.abs(mk.col[KNOWN[1]][pos + H] - t[SHOCK_COLS[1]].to_numpy()).max()
    check("타깃 h=5 값 = train_table 정답(5일 뒤 수익률)", d_y < 1e-6, f"{len(t):,}행, 최대 차이 {d_y:.2e}")
    check("금리 충격 h=5 값 = train_table shock_us10y_fwd5d_bp", d_r < 1e-6, f"최대 차이 {d_r:.2e}")
    check("환율 충격 h=5 값 = train_table shock_usdkrw_fwd5d_pct", d_f < 1e-5, f"최대 차이 {d_f:.2e}")

    # 3) 충격 경로
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in rng.choice(pos, 40, replace=False):
        for shape in ("linear", "front", "back"):
            rb, fp = float(rng.uniform(-30, 30)), float(rng.uniform(-4, 4))
            fut = mk.future_ramp(int(i), "x", rb, fp, shape)
            worst = max(worst, abs(fut[KNOWN[0]].iloc[-1] - rb), abs(fut[KNOWN[1]].iloc[-1] - fp))
    check("충격 경로의 h=5 값 = 입력한 충격 (선형·첫날·마지막날)", worst < 1e-9, f"최대 차이 {worst:.2e}")
    i = int(pos[2000])
    real = mk.future_realized(i, "x")
    ramp = mk.future_ramp(i, "x", float(t[SHOCK_COLS[0]].iloc[2000]), float(t[SHOCK_COLS[1]].iloc[2000]), "linear")
    check("실제 경로와 선형 경로가 h=5 에서 일치", abs(real[KNOWN[0]].iloc[-1] - ramp[KNOWN[0]].iloc[-1]) < 1e-6
          and abs(real[KNOWN[1]].iloc[-1] - ramp[KNOWN[1]].iloc[-1]) < 1e-6)
    base = mk.future_ramp(i, "x", 0.0, 0.0, "linear")
    check("BASE(충격 0) h=5 값 = 0", abs(base[KNOWN[0]].iloc[-1]) < 1e-9 and abs(base[KNOWN[1]].iloc[-1]) < 1e-9)

    # 4) 입력표 형식
    specs = [dict(id=str(mk.date(int(p)).date()), origin=int(p), rate_bp=0.0, fx_pct=0.0) for p in pos[3000:3005]]
    for variant in VARIANTS:
        ctx, fut = build_requests(mk, specs, 512, variant)
        g = ctx.groupby("id")["timestamp"]
        regular = all((x.diff().dropna() == pd.Timedelta(days=1)).all() for _, x in g)
        check(f"[{variant}] 컨텍스트 날짜가 하루 간격으로 이어짐", regular, f"열 {list(ctx.columns)}")
        if fut is not None:
            first_fut = fut.groupby("id")["timestamp"].first()
            last_ctx = g.last()
            seam = ((first_fut - last_ctx) == pd.Timedelta(days=1)).all() and fut.groupby("id").size().eq(H).all()
            check(f"[{variant}] future_df 가 컨텍스트 바로 다음 {H}일, 열 {[c for c in fut.columns if c not in ('id', 'timestamp')]}", seam)
    try:
        from chronos.df_utils import validate_and_normalize_df
        for variant in VARIANTS:
            ctx, fut = build_requests(mk, specs, 512, variant)
            validate_and_normalize_df(df=ctx, future_df=fut, target_columns=[TARGET], prediction_length=H,
                                      id_column="id", timestamp_column="timestamp")
        check("chronos 공식 입력 검사(validate_and_normalize_df) 통과 (모든 입력 구성)", True)
    except ImportError:
        print("[건너뜀] chronos 가 설치돼 있지 않아 공식 입력 검사는 생략 (설치 후 다시 실행하면 검사합니다)")
    except Exception as e:                                                # noqa: BLE001
        check("chronos 공식 입력 검사(validate_and_normalize_df) 통과", False, f"{type(e).__name__}: {e}")

    # 5) 모의 예측기로 전체 흐름
    rows = t[t.split == "valid"].iloc[:200]
    specs = [dict(id=str(d.date()), origin=mk.pos(d), rate_bp=float(a), fx_pct=float(b))
             for d, a, b in zip(rows["date"], rows[SHOCK_COLS[0]], rows[SHOCK_COLS[1]])]
    res = forecast(FakePipeline(), mk, specs, 256, "full", batch_items=50, log=None)
    s5 = res[res.step == H].set_index("id")
    q = s5.loc[[sp["id"] for sp in specs], [str(x) for x in TAUS]].to_numpy()
    dev = t[t.split == "train"]
    m = {**baselines(dev, rows), "FAKE": q}
    vix_hi = dev["vix"].quantile(2 / 3)
    table = score(m, rows[Y_RET], rows["vix"], vix_hi)
    check("모의 예측 → 지표 계산까지 끝까지 실행", len(table) == 3 and table["pinball"].notna().all())
    for variant in VARIANTS:
        r = forecast(FakePipeline(), mk, specs[:20], 256, variant, batch_items=20, log=None)
        check(f"[{variant}] 모의 예측이 20일 × {H}단계 모두 나옴",
              len(r) == 20 * H and r[[str(x) for x in TAUS]].notna().all().all())

    # 5-2) 구간 보정: 일부러 좁게 만든 예측을 보정하면 하단·상단 이탈이 각각 10% 가 되는가
    rg = np.random.default_rng(1)
    n = 4000
    sd = rg.uniform(0.8, 2.0, n)
    yy = rg.normal(0, 1.4 * sd)                                  # 실제 변동은 예측(sd)의 1.4배
    qq = np.column_stack([-1.2816 * sd, np.zeros(n), 1.2816 * sd])
    vx = rg.uniform(10, 40, n)
    cal = fit_calibration(yy, qq, vx, 25.0)
    for mode in ("global", "by_vix"):
        qc = apply_calibration(qq, vx, cal, mode)
        lo, up = float(np.mean(yy < qc[:, 0])), float(np.mean(yy > qc[:, 2]))
        check(f"구간 보정({mode}) 후 하단·상단 이탈이 각각 10% 근처", abs(lo - .1) < .01 and abs(up - .1) < .01,
              f"하단 {lo:.3f}, 상단 {up:.3f}")
    check("구간 보정은 중앙값을 바꾸지 않고 배율이 약 1.4", np.allclose(qc[:, 1], qq[:, 1])
          and abs(cal["global"]["k_lo"] - 1.4) < .1, f"k_lo={cal['global']['k_lo']:.2f}, k_hi={cal['global']['k_hi']:.2f}")
    ql = apply_calibration(qq, vx, cal, "lower")
    check("'하단만' 보정은 상단·중앙값을 그대로 두고 하단 이탈만 10%로 맞춤",
          np.allclose(ql[:, 1:], qq[:, 1:]) and abs(float(np.mean(yy < ql[:, 0])) - .1) < .01,
          f"하단 이탈 {float(np.mean(yy < ql[:, 0])):.3f}")

    # 5-3) 파인튜닝 입력: 학습 시계열과 검증 시계열 구성
    ser = s[(s["date"] >= t["date"].min()) & (s["date"] <= pd.Timestamp("2018-12-31"))].reset_index(drop=True)
    n_fit = int((ser["date"] <= pd.Timestamp("2016-12-31")).sum())
    for variant in ("select", "no_cov"):
        tr, va = make_items(ser, variant, n_fit, 512)
        past_cols, known_cols = VARIANTS[variant]
        ok_shape = len(tr["target"]) == n_fit and all(len(v["target"]) <= 512 + H for v in va) \
            and set(tr.get("past_covariates", {})) == set(past_cols) | set(known_cols)
        check(f"[{variant}] 파인튜닝 학습 시계열 {n_fit:,}일 + 검증 {len(va)}개 구성", ok_shape and 90 <= len(va) <= 110)
    tr, va = make_items(ser, "select", n_fit, 512)
    j = 7
    e = n_fit + H - 1 + j * H
    origin = ser["date"].iloc[e - H]
    y_true = float(t.loc[t["date"] == origin, Y_RET].iloc[0])
    shock_true = float(t.loc[t["date"] == origin, SHOCK_COLS[0]].iloc[0])
    check("검증 시계열의 마지막 값 = 기준일의 정답·충격 (타깃·충격 정렬)",
          abs(va[j]["target"][-1] - y_true) < 1e-4 and abs(va[j]["past_covariates"][KNOWN[0]][-1] - shock_true) < 1e-3,
          f"기준일 {origin:%Y-%m-%d}: 정답 {y_true:.3f}")
    check("검증 구간의 정답이 학습 시계열에 들어있지 않음", ser["date"].iloc[e - H + 1] > pd.Timestamp("2016-12-31"))

    # 6) 지표 계산이 kNN 결과와 같은 숫자를 내는가
    if KNN_TEST_PRED.exists():
        k = pd.read_csv(KNN_TEST_PRED, parse_dates=["date"]).set_index("date")
        te = t[t.split == "test"]
        dev2 = t[t.split != "test"]
        q = k.loc[te.date, ["pred_ret_q10", "pred_ret_q50", "pred_ret_q90"]].to_numpy()
        vix_hi = t.loc[t.split == "train", "vix"].quantile(2 / 3)
        tab = score({**baselines(dev2, te), "kNN": q}, te[Y_RET], te["vix"], vix_hi).set_index("model")
        card = json.loads((KNN / "models" / "model_card.json").read_text(encoding="utf-8"))["test_result"]["FINAL_knn_calibrated_ret"]
        check("kNN test pinball 이 model_card 와 같음", abs(tab.loc["kNN", "pinball"] - card["pinball"]) < 1e-4,
              f"{tab.loc['kNN', 'pinball']:.4f} vs {card['pinball']:.4f}")
        check("kNN test 80% 포함률이 model_card 와 같음", abs(tab.loc["kNN", "coverage80"] - card["coverage80"]) < 1e-4)
        check("kNN skill_vs_M1 이 model_card 와 같음", abs(tab.loc["kNN", "skill_vs_M1"] - card["skill_vs_M1"]) < 1e-3,
              f"{tab.loc['kNN', 'skill_vs_M1']:.4f} vs {card['skill_vs_M1']:.4f}")
    print("\n" + ("모든 점검 통과" if ok_all else "실패한 항목이 있습니다. 위 [실패] 줄을 확인하세요."))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
