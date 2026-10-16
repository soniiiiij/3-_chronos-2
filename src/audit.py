# -*- coding: utf-8 -*-
"""
감사용 자료 만들기: 평가 결과를 '날짜별 원자료'와 '검증 가능한 표'로 풀어서 results/audit/ 에 저장한다.

  만드는 것
    audit_predictions_valid.csv  날짜별 정답·기준/목표 종가와 날짜·충격·VIX·모든 모델의 q10/q50/q90 (한 파일)
    audit_sample_rows.csv        예측 3건(무작위, 시드 고정) + 정상·최대 급락·최대 급반등 사례
    audit_by_year.csv            모델×연도: 표본 수, 80% 포함률, 하단·상단 이탈률, 구간 폭
    audit_pinball_by_tau.csv     모델별 분위수(0.1/0.5/0.9) 따로 본 pinball 과 평균
    audit_events.csv             5일 수익률 −3% / −5% 이하 사건 수(학습·valid·test)와 valid 에서 각 모델 하단(q10) 밖으로 나간 건수
    audit_tables.md              위 표를 한 문서로

  ※ 새 모델을 돌리지 않고, results/ 에 있는 eval_valid_*_predictions.csv 만 읽는다. 학습표·원자료는 읽기만 한다.
  ※ 경보 규칙을 만든 적이 없으므로 '포착률·오경보율'은 계산하지 않는다 (사건 수와 하단 이탈 건수만 보여 준다).

실행:  python src/audit.py
"""
import re

import numpy as np
import pandas as pd

from common import (RESULTS, PROJECT, VARIANTS, Y_RET, SHOCK_COLS, TAUS, load_truth, baselines, pinball, sigma_5d)

OUT = RESULTS / "audit"
RAW_XLSX = PROJECT / "Datasets" / "market_features_daily.xlsx"
PAT = re.compile(r"^eval_valid_(.+)_(" + "|".join(sorted(VARIANTS, key=len, reverse=True)) + r")_L(\d+)_predictions\.csv$")
QC = ["pred_ret_q10", "pred_ret_q50", "pred_ret_q90"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    truth = load_truth()
    dev = truth[truth.split == "train"]
    rows = truth[truth.split == "valid"].reset_index(drop=True)
    vix_hi = float(dev["vix"].quantile(2 / 3))

    # ── 원자료: 코스피 종가, 미국 자료의 참조 날짜
    raw = pd.read_excel(RAW_XLSX, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raw["sp500_reference_date"] = pd.to_datetime(raw["sp500_reference_date"])
    pos = {d: i for i, d in enumerate(raw["date"])}
    i0 = rows["date"].map(pos).to_numpy()
    base = pd.DataFrame({
        "origin_date": rows["date"].dt.date, "forecast_time_KST": "18:00 (코스피 종가 15:30 이후)",
        "us_data_date": raw["sp500_reference_date"].to_numpy()[i0],
        "base_close_date": rows["date"].dt.date, "base_close": raw["kospi"].to_numpy()[i0],
        "target_close_date": raw["date"].to_numpy()[i0 + 5], "target_close": raw["kospi"].to_numpy()[i0 + 5],
        "y_ret_pct": rows[Y_RET].to_numpy(), SHOCK_COLS[0]: rows[SHOCK_COLS[0]].to_numpy(),
        SHOCK_COLS[1]: rows[SHOCK_COLS[1]].to_numpy(), "vix_asof_origin": rows["vix"].to_numpy(),
        "high_vix_flag": (rows["vix"].to_numpy() > vix_hi)})
    base["y_recomputed_pct"] = 100 * (base["target_close"] / base["base_close"] - 1)
    base["target_close_date"] = pd.to_datetime(base["target_close_date"]).dt.date
    base["us_data_date"] = pd.to_datetime(base["us_data_date"]).dt.date
    diff = float(np.max(np.abs(base["y_recomputed_pct"] - base["y_ret_pct"])))
    ok_log = float(np.max(np.abs(100 * np.log(base["target_close"] / base["base_close"]) - base["y_ret_pct"])))
    print(f"[정답 확인] 원자료 종가로 단순수익률 100×(종가[t+5]/종가[t]−1)을 다시 계산: 학습표 정답과 최대 차이 {diff:.2e} %p "
          f"(로그수익률이었다면 최대 차이 {ok_log:.2f} %p)")
    print(f"[VIX 기준] 고VIX = 기준일 행의 vix > {vix_hi:.2f}  (학습 구간 {dev['date'].min():%Y-%m-%d}~{dev['date'].max():%Y-%m-%d} VIX 의 상위 1/3 경계. "
          f"기준일 행의 vix 는 직전 미국 장 종가라 예측 시점에 알려진 값)")

    # ── 모델별 예측 모으기
    models = {k: np.asarray(v, float) for k, v in baselines(dev, rows).items()}
    for f in sorted(RESULTS.glob("eval_valid_*_predictions.csv")):
        m = PAT.match(f.name)
        if not m:
            continue
        tag, variant, L = m.groups()
        if tag.startswith("fake"):                       # 모의 예측기 결과는 감사 대상이 아님
            continue
        d = pd.read_csv(f, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if len(d) != len(rows) or not (d["date"].to_numpy() == rows["date"].to_numpy()).all():
            print(f"  제외(날짜 불일치/부분 실행): {f.name}")
            continue
        assert np.allclose(d["y_ret"].to_numpy(), rows[Y_RET].to_numpy(), atol=1e-6), f"{f.name}: 정답이 학습표와 다름"
        models[f"{tag}|{variant}|L{L}"] = np.sort(d[QC].to_numpy(), axis=1)
    names = list(models)
    print(f"[예측 파일] 같은 {len(rows)}건으로 확인된 모델: " + ", ".join(names))

    wide = base.copy()
    for k, q in models.items():
        short = k.replace("amazon_chronos-2|", "chronos2_").replace("|L512", "").replace("|", "_")
        for j, c in enumerate(("q10", "q50", "q90")):
            wide[f"{short}__{c}"] = q[:, j]
    wide.to_csv(OUT / "audit_predictions_valid.csv", index=False, encoding="utf-8-sig")

    y = rows[Y_RET].to_numpy()
    yr = rows["date"].dt.year.to_numpy()

    # ── 연도별
    recs = []
    for k, q in models.items():
        for yy in sorted(set(yr)):
            m = yr == yy
            ins = (y[m] >= q[m, 0]) & (y[m] <= q[m, 2])
            recs.append({"model": k, "year": int(yy), "n": int(m.sum()), "coverage80": ins.mean(),
                         "below10": np.mean(y[m] < q[m, 0]), "above10": np.mean(y[m] > q[m, 2]),
                         "width80": np.mean(q[m, 2] - q[m, 0]), "y_std": y[m].std(), "vix_mean": rows["vix"].to_numpy()[m].mean()})
    by_year = pd.DataFrame(recs)
    by_year.to_csv(OUT / "audit_by_year.csv", index=False, encoding="utf-8-sig")

    # ── 분위수별 pinball
    recs = []
    for k, q in models.items():
        p = [float(np.mean(pinball(y, q[:, j], tau))) for j, tau in enumerate(TAUS)]
        recs.append({"model": k, "pinball_tau0.1": p[0], "pinball_tau0.5": p[1], "pinball_tau0.9": p[2], "평균(=표의 pinball)": float(np.mean(p))})
    by_tau = pd.DataFrame(recs)
    ref = by_tau.set_index("model").loc["M1_unconditional"]
    for c in ("pinball_tau0.1", "pinball_tau0.5", "pinball_tau0.9", "평균(=표의 pinball)"):
        by_tau[f"개선율_vs_M1({c.split('_')[-1][:6]})"] = 1 - by_tau[c] / ref[c]
    by_tau.to_csv(OUT / "audit_pinball_by_tau.csv", index=False, encoding="utf-8-sig")

    # ── 사건 수 (−3%, −5% 이하)
    recs = []
    for sp in ("train", "valid"):                      # test 는 마지막에 한 번만 보기로 했으므로 사건 수도 보지 않는다
        yy = truth.loc[truth.split == sp, Y_RET]
        recs.append({"구간": sp, "표본": len(yy), "5일수익률<=-3%": int((yy <= -3).sum()), "5일수익률<=-5%": int((yy <= -5).sum())})
    ev = pd.DataFrame(recs)
    ev["<=-3% 비율"] = ev["5일수익률<=-3%"] / ev["표본"]
    ev["<=-5% 비율"] = ev["5일수익률<=-5%"] / ev["표본"]
    ev.to_csv(OUT / "audit_events.csv", index=False, encoding="utf-8-sig")
    recs = []
    for th in (-3, -5):
        e = y <= th
        for k, q in models.items():
            recs.append({"사건": f"5일수익률<={th}%", "사건 수": int(e.sum()), "model": k,
                         "정답이 q10 아래(하단 밖)": int((y[e] < q[e, 0]).sum()), "정답이 q10~q90 안": int(((y[e] >= q[e, 0]) & (y[e] <= q[e, 2])).sum()),
                         "q10 자체가 임계값 이하인 날(전체 983일 중)": int((q[:, 0] <= th).sum()),
                         "그중 실제 사건": int(((q[:, 0] <= th) & e).sum())})
    ev2 = pd.DataFrame(recs)
    ev2.to_csv(OUT / "audit_events_by_model.csv", index=False, encoding="utf-8-sig")

    # ── 예시 행
    rng = np.random.default_rng(0)
    pick = list(rng.choice(len(rows), 3, replace=False))
    picks = [("무작위1(시드0)", pick[0]), ("무작위2(시드0)", pick[1]), ("무작위3(시드0)", pick[2]),
             ("정상(정답이 중앙값에 가장 가까운 날)", int(np.argmin(np.abs(y - np.median(y))))),
             ("최대 급락(valid 최소 정답)", int(np.argmin(y))), ("최대 급반등(valid 최대 정답)", int(np.argmax(y)))]
    key_models = [k for k in names if k.startswith("amazon_chronos-2|full") or k.startswith("amazon_chronos-2|no_cov") or k in ("M1_unconditional", "M2_volscale")]
    recs = []
    for lab, i in picks:
        r = {"사례": lab, "예측 기준시각": f"{base.origin_date[i]} 18:00 KST", "입력 마지막 시각": f"한국 {base.origin_date[i]} 장 마감(15:30)까지 + 미국 {base.us_data_date[i]} 종가",
             "기준 종가(날짜·가격)": f"{base.base_close_date[i]} · {base.base_close[i]:,.2f}", "목표 종가(날짜·가격)": f"{base.target_close_date[i]} · {base.target_close[i]:,.2f}",
             "실제 y(%)": round(float(y[i]), 2), "금리충격(bp)": float(base[SHOCK_COLS[0]][i]), "환율충격(%)": round(float(base[SHOCK_COLS[1]][i]), 2),
             "VIX": float(base.vix_asof_origin[i])}
        for k in key_models:
            r[f"{k} q10/q50/q90"] = " / ".join(f"{v:.2f}" for v in models[k][i])
        recs.append(r)
    samples = pd.DataFrame(recs)
    samples.to_csv(OUT / "audit_sample_rows.csv", index=False, encoding="utf-8-sig")

    # ── 한 문서로
    def md(df, floatfmt=3):
        return df.round(floatfmt).to_markdown(index=False) if hasattr(df, "to_markdown") else df.round(floatfmt).to_string(index=False)
    try:
        import tabulate  # noqa: F401
        has_md = True
    except Exception:
        has_md = False
    show = (lambda df, n=3: md(df, n)) if has_md else (lambda df, n=3: "```\n" + df.round(n).to_string(index=False) + "\n```")
    doc = ["# 감사용 표 (audit.py 가 results/ 의 예측 파일에서 만든 것)\n",
           f"- 평가 구간: valid {rows['date'].min():%Y-%m-%d} ~ {rows['date'].max():%Y-%m-%d}, {len(rows)}건 (매 거래일). 고VIX 기준 vix > {vix_hi:.2f}\n",
           f"- 정답 재계산: 원자료 종가로 계산한 단순수익률(%)과 학습표 정답의 최대 차이 {diff:.1e}\n",
           "## 연도별\n", show(by_year), "\n## 분위수별 pinball\n", show(by_tau), "\n## 5일 수익률 사건 수\n", show(ev),
           "\n## 사건에서 각 모델의 하단(q10) 밖 건수\n", show(ev2, 0), "\n## 예시 행\n", show(samples, 2)]
    (OUT / "audit_tables.md").write_text("\n".join(doc), encoding="utf-8")

    print("\n=== 연도별 (80% 포함률 / 하단·상단 이탈 / 폭) ===")
    print(by_year[by_year.model.str.contains("full|no_cov|M1|M2|select|all")].round(3).to_string(index=False))
    print("\n=== 분위수별 pinball ==="); print(by_tau.round(3).to_string(index=False))
    print("\n=== 5일 수익률 사건 수 ==="); print(ev.round(3).to_string(index=False))
    print("\n=== 사건에서 하단(q10) 밖 ==="); print(ev2.to_string(index=False))
    print("\n=== 예시 ==="); print(samples.T.to_string(header=False))
    print(f"\n저장: results/audit/ ({len(list(OUT.glob('*')))}개 파일)")


if __name__ == "__main__":
    main()
