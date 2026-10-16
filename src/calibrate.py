# -*- coding: utf-8 -*-
"""
구간 보정: "80% 범위"라고 보여 주는 하단 10% ~ 상단 90% 가 실제로 80% 가 되도록 범위를 늘이거나 줄인다.

  왜    : 제로샷 Chronos-2 는 범위가 좁아서 정답이 구간 안에 들어오는 비율이 약 71% 였다 (목표 80%).
  원리  : 중앙값은 그대로 두고, 중앙값에서 하단·상단까지의 거리에 배율을 곱한다.
          배율 = 보정 기간의 정답으로 구한, 하단 이탈과 상단 이탈이 각각 10% 가 되는 값.
          예측 범위가 넓은 날은 더 넓게, 좁은 날은 덜 넓게 보정된다. (--by-vix: VIX 낮은 날/높은 날 따로)
  검증  : 같은 자료로 맞추고 같은 자료로 채점하면 점수가 부풀려진다. 그래서
          앞 기간(기본 2019~2020)으로 배율을 구하고, 보지 못한 뒤 기간(2021~2022)에서 성적이 실제로 좋아졌는지 본다.
  저장  : valid 전체(2019~2022)로 구한 배율을 results/calib_<이름>.json 에 저장한다.
          → evaluate.py --calib, predict_scenario.py --calib 에서 쓴다. (test 는 그 뒤 기간이라 그대로 적용 가능)
          단, 보정 파일은 '그 모델'의 예측으로 만든 것이다. 다른 모델(예: 다시 학습한 최종 모델)에는 맞지 않을 수 있다.

실행 예 (evaluate.py --split valid 가 만든 예측 파일을 넣는다)
  python src/calibrate.py --pred results/eval_valid_amazon_chronos-2_select_L512_predictions.csv
  python src/calibrate.py --pred results/eval_valid_amazon_chronos-2_all_L512_predictions.csv results/eval_valid_m3_select_select_L512_predictions.csv
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, Y_RET, load_truth, baselines, score, fit_calibration, apply_calibration

QC = ["pred_ret_q10", "pred_ret_q50", "pred_ret_q90"]


def main():
    ap = argparse.ArgumentParser(description="구간 보정 (앞 기간으로 배율을 구하고 뒤 기간에서 확인)")
    ap.add_argument("--pred", nargs="+", required=True, help="evaluate.py --split valid 가 만든 *_predictions.csv (여러 개 가능)")
    ap.add_argument("--until", default="2020-12-31", help="이 날짜까지를 보정 기간, 이후를 확인 기간으로 (기본 2020-12-31)")
    ap.add_argument("--alpha", type=float, default=0.10, help="하단·상단 각각의 목표 이탈률 (기본 0.10)")
    ap.add_argument("--embargo", type=int, default=5,
                    help="보정 기간 끝의 N거래일은 배율 계산에서 뺀다. 그 날들의 5일 정답은 확인 기간이 시작된 뒤에야 확정되므로 "
                         "시점 분리를 엄격하게 하려는 것 (기본 5)")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30)

    truth = load_truth()
    dev = truth[truth.split == "train"]
    valid_dates = set(truth.loc[truth.split == "valid", "date"])
    vix_hi = dev["vix"].quantile(2 / 3)
    until = pd.Timestamp(a.until)

    for path in a.pred:
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        label = Path(path).name.replace("_predictions.csv", "").replace("eval_valid_", "")
        if not set(df["date"]) <= valid_dates:
            raise SystemExit(f"{Path(path).name}: valid(2019~2022) 예측 파일이 아닙니다. test 자료로 보정 배율을 만들면 안 됩니다.")
        m = df["date"] <= until
        if m.sum() < 200 or (~m).sum() < 200:
            raise SystemExit(f"{Path(path).name}: 보정 기간 {int(m.sum())}일 / 확인 기간 {int((~m).sum())}일 — 각각 200일 이상 필요합니다. "
                             f"전체 valid 예측(983일)을 쓰고 있는지, --until 을 확인하세요.")
        C, T = df[m], df[~m].reset_index(drop=True)
        if a.embargo:                                     # 보정 기간 마지막 N일의 정답은 확인 기간 첫 며칠 사이에 확정 → 제외
            C = C.iloc[:-a.embargo]
        cal = fit_calibration(C["y_ret"], C[QC].to_numpy(), C["vix"], vix_hi, a.alpha)

        rows = truth.set_index("date").loc[T["date"]].reset_index()
        assert np.allclose(rows[Y_RET].to_numpy(), T["y_ret"].to_numpy(), atol=1e-6), "예측 파일의 정답이 학습표와 다릅니다"
        qT = T[QC].to_numpy()
        models = {**baselines(dev, rows), "원래": qT,
                  "보정(전체 배율)": apply_calibration(qT, T["vix"], cal, "global"),
                  "보정(하단만)": apply_calibration(qT, T["vix"], cal, "lower"),
                  "보정(VIX 구간별)": apply_calibration(qT, T["vix"], cal, "by_vix")}
        M = score(models, T["y_ret"], T["vix"], vix_hi)

        print(f"\n━━ {label}")
        g, lo, hi = cal["global"], cal["by_vix"]["low"], cal["by_vix"]["high"]
        print(f"보정 기간 {C['date'].min():%Y-%m-%d}~{C['date'].max():%Y-%m-%d} ({len(C)}일)에서 구한 배율 (1보다 크면 원래 범위가 좁았다는 뜻)"
              + (f"  ※ 정답이 확인 기간 이후에 확정되는 마지막 {a.embargo}거래일은 뺌" if a.embargo else ""))
        print(f"  전체       하단 ×{g['k_lo']:.2f}  상단 ×{g['k_hi']:.2f}")
        print(f"  VIX 낮은 날 하단 ×{lo['k_lo']:.2f}  상단 ×{lo['k_hi']:.2f}  ({lo['n']}일)")
        print(f"  VIX 높은 날 하단 ×{hi['k_lo']:.2f}  상단 ×{hi['k_hi']:.2f}  ({hi['n']}일)")
        print(f"\n확인 기간 {T['date'].min():%Y-%m-%d}~{T['date'].max():%Y-%m-%d} ({len(T)}일, 서로 독립인 구간은 약 {len(T) // 5}개)의 성적 "
              f"— 목표: 80% 포함률 0.80 / 하단·상단 이탈 각 0.10")
        show = ["model", "n", "pinball", "skill_vs_M1", "ci_vs_M1", "skill_vs_M2", "coverage80", "below10", "above10",
                "coverage80_vix_high", "width80"]
        print(M[show].round(3).to_string(index=False))

        yy = df.assign(year=df["date"].dt.year, inside=(df["y_ret"] >= df[QC[0]]) & (df["y_ret"] <= df[QC[2]]),
                       below=df["y_ret"] < df[QC[0]], above=df["y_ret"] > df[QC[2]])
        yt = yy.groupby("year").agg(일수=("inside", "size"), 포함률=("inside", "mean"), 하단이탈=("below", "mean"), 상단이탈=("above", "mean"))
        print("\n(참고) 원래 예측의 연도별 성적 — 어느 해에서 범위가 모자랐는지:")
        print(yt.round(3).to_string())

        final = fit_calibration(df["y_ret"], df[QC].to_numpy(), df["vix"], vix_hi, a.alpha)
        final.update(source=Path(path).name, fit_start=str(df["date"].min().date()), fit_end=str(df["date"].max().date()),
                     created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        out = RESULTS / f"calib_{label}.json"
        out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        f = final["global"]
        print(f"\n저장: results/{out.name}   (valid 전체 {final['n']}일로 구한 최종 배율: 하단 ×{f['k_lo']:.2f}, 상단 ×{f['k_hi']:.2f})")
        print(f"  적용:  python src/evaluate.py --split test --final --variants <구성> --calib results/{out.name}   (마지막에 한 번만)")

    print("\n읽는 법: '원래'보다 '보정' 행의 80% 포함률이 0.80 에 가까워지고 pinball 이 같거나 낮으면 보정이 통한 것입니다.")
    print("※ 확인 기간이 짧아(독립 구간 약 100개) 작은 차이는 우연일 수 있습니다. 전체 배율·하단만·VIX 구간별 중 단순한 쪽을 우선 쓰세요.")
    print("※ 보정 기간(2019~2020)에 코로나 급락이 들어 있어 배율이 크게 나오고, 조용한 해(2021~)에는 상단이 과하게 넓어질 수 있습니다.")
    print("   '하단만'은 손실 쪽 범위만 넓히므로 이 문제가 작습니다. (서비스에서 보여줄 위험 범위가 하단이기도 합니다.)")


if __name__ == "__main__":
    main()
