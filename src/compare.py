# -*- coding: utf-8 -*-
"""
results/ 에 쌓인 모든 평가 결과(eval_<split>_*_predictions.csv)를 같은 기준으로 다시 채점해서 한 표로 비교한다.

  - 예측 파일이 구성(all/select/full ...)·모델·컨텍스트마다 따로 저장되므로, 여러 번 나눠 돌린 결과도 한 번에 비교된다.
  - 기준선(과거 분포 M1, 변동성 스케일 M2)과 지표는 evaluate.py 와 같은 방식으로 계산한다.
  - 행 수가 다른 파일(--limit 로 일부만 돌린 것)은 제외한다.

실행 예
  python src/compare.py                  # valid 결과 전부
  python src/compare.py --split test     # test 결과 (기존 kNN 행 포함)
  python src/compare.py --ref past_only  # 충격 효과: 같은 날짜·정답에서 past_only 대비 차이와 짝지은 95% 구간
"""
import argparse
import re

import numpy as np
import pandas as pd

from common import (RESULTS, VARIANTS, Y_RET, KNN_TEST_PRED, load_truth, baselines, score, range_metrics,
                    block_bootstrap_diff)

PAT_TMPL = r"^eval_{split}_(.+)_({variants})_L(\d+)_predictions\.csv$"


def main():
    ap = argparse.ArgumentParser(description="모든 평가 결과를 한 표로 비교")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--sort", default="pinball", help="정렬 기준 열 (기본 pinball, 낮은 순)")
    ap.add_argument("--ref", default=None, help="짝지은 비교의 기준 입력 구성 (예: past_only). 충격 효과는 --ref past_only 로 full 과 비교")
    a = ap.parse_args()
    pat = re.compile(PAT_TMPL.format(split=a.split, variants="|".join(sorted(VARIANTS, key=len, reverse=True))))

    truth = load_truth()
    rows = truth[truth.split == a.split].reset_index(drop=True)
    dev = truth[truth.split == "train"] if a.split == "valid" else truth[truth.split != "test"]
    models = baselines(dev, rows)
    skipped = []
    for f in sorted(RESULTS.glob(f"eval_{a.split}_*_predictions.csv")):
        mt = pat.match(f.name)
        if not mt:
            continue
        tag, variant, L = mt.groups()
        d = pd.read_csv(f, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if len(d) != len(rows) or not (d["date"].to_numpy() == rows["date"].to_numpy()).all():
            skipped.append(f"{f.name} ({len(d)}일)")
            continue
        models[f"{tag}|{variant}|L{L}"] = d[["pred_ret_q10", "pred_ret_q50", "pred_ret_q90"]].to_numpy()
    if len(models) == 2:
        raise SystemExit(f"results/ 에서 {a.split} 평가 결과를 찾지 못했습니다. 먼저 evaluate.py --split {a.split} 를 실행하세요.")
    if a.split == "test" and KNN_TEST_PRED.exists():
        k = pd.read_csv(KNN_TEST_PRED, parse_dates=["date"]).set_index("date").loc[rows["date"]]
        models["기존_kNN(참고)"] = k[["pred_ret_q10", "pred_ret_q50", "pred_ret_q90"]].to_numpy()

    vix_hi = truth.loc[truth.split == "train", "vix"].quantile(2 / 3)
    M = score(models, rows[Y_RET], rows["vix"], vix_hi)
    M = M.sort_values(a.sort).reset_index(drop=True)
    show = ["model", "pinball", "skill_vs_M1", "ci_vs_M1", "skill_vs_M2", "ci_vs_M2", "coverage80", "below10", "above10",
            "coverage80_vix_high", "width80", "mae_median"]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"=== {a.split} 비교 ({len(rows):,}일, 이름 = 모델|입력 구성|컨텍스트) — 목표: 80% 포함률 0.80 / 하단·상단 이탈 각 0.10 ===")
    print(M[show].round(3).to_string(index=False))
    if skipped:
        print("\n제외한 파일(행 수 또는 날짜가 달라 비교 불가): " + ", ".join(skipped))
    out = RESULTS / f"compare_{a.split}.csv"
    M.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: results/{out.name}")

    if a.ref:                                   # 짝지은 비교: 같은 날짜·같은 정답에서 기준 구성(--ref) 대비 차이
        keys = [k for k in models if f"|{a.ref}|" in k or k == a.ref]
        if not keys:
            raise SystemExit(f"--ref {a.ref} 에 해당하는 결과가 없습니다. 먼저 evaluate.py --variants {a.ref} 를 실행하세요.")
        ref = keys[0]
        y = rows[Y_RET].to_numpy()
        loss = {k: range_metrics(y, *np.sort(np.asarray(q, float), axis=1).T)[1] for k, q in models.items()}
        mae = {k: float(np.mean(np.abs(y - np.sort(np.asarray(q, float), axis=1)[:, 1]))) for k, q in models.items()}
        Mi = M.set_index("model")
        recs = []
        for k in models:
            if k == ref:
                continue
            (dlo, dhi), (slo, shi) = block_bootstrap_diff(loss[k], loss[ref])
            recs.append({"model": k, "pinball": Mi.loc[k, "pinball"], "pinball_차이(모델−기준)": loss[k].mean() - loss[ref].mean(),
                         "차이_95%구간": f"{dlo:+.3f}~{dhi:+.3f}", "skill_vs_기준": 1 - loss[k].mean() / loss[ref].mean(),
                         "skill_95%구간": f"{slo:+.3f}~{shi:+.3f}", "mae_median": mae[k], "mae_차이": mae[k] - mae[ref],
                         "coverage80": Mi.loc[k, "coverage80"], "width80": Mi.loc[k, "width80"],
                         "below10": Mi.loc[k, "below10"], "above10": Mi.loc[k, "above10"]})
        P = pd.DataFrame(recs).sort_values("pinball").reset_index(drop=True)
        print(f"\n=== 짝지은 비교: 기준 = {ref}  (pinball {Mi.loc[ref, 'pinball']:.3f}, mae {mae[ref]:.3f}, "
              f"포함률 {Mi.loc[ref, 'coverage80']:.3f}, 폭 {Mi.loc[ref, 'width80']:.2f}) ===")
        print("차이가 음수 = 기준보다 좋음. 95% 구간은 같은 날짜를 함께 뽑는 20거래일 블록 부트스트랩(2,000회, 시드 0). 0 을 포함하면 차이를 확신할 수 없음.")
        print(P.round(3).to_string(index=False))
        fp = RESULTS / f"compare_{a.split}_paired_{a.ref}.csv"
        P.to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"\n저장: results/{fp.name}")


if __name__ == "__main__":
    main()
