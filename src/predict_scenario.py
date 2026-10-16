# -*- coding: utf-8 -*-
"""
사용자 입력 2개(금리 충격, 환율 충격)로 BASE / STRESS 를 계산한다 (기존 kNN 의 step7_predict.py 에 해당).

  BASE   = 같은 시장 상태 + 충격 (0bp, 0%)
  STRESS = 같은 시장 상태 + 사용자 충격 (rate bp, fx %)   ← 5거래일에 걸쳐 --shape 모양으로 들어온다고 가정
  출력   : 5일 뒤 수익률의 하단 10% / 중앙 / 상단 90% (BASE, STRESS, 차이), 1~5일차 경로별 분위수

  ※ 이 모델은 '5일 중 최저 수익률'과 '하락 확률'을 내지 않는다. (단계별 분위수만 나오기 때문)
     두 값이 필요하면 기존 kNN 쪽(model_training/src/step7_predict.py)을 함께 쓴다.
  ※ --curve 를 붙이면 충격 크기를 바꿔 가며 결과가 말이 되게 움직이는지(반응 곡선)를 표와 그림으로 저장한다.

실행 예
  python src/predict_scenario.py --model amazon/chronos-2 --rate 20 --fx 0
  python src/predict_scenario.py --model models/lora_upto2022/finetuned-ckpt --rate 0 --fx 2 --date 2022-09-26
  python src/predict_scenario.py --rate 20 --curve
  python src/predict_scenario.py --fake --rate 20          # chronos 없이 배관만 점검
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from common import (H, TAUS, KNOWN, RESULTS, VARIANTS, Market, load_state, forecast, FakePipeline, load_pipeline,
                    read_meta, apply_calibration, load_calibration)


def run(pipe, mk, i, specs_rf, L, variant, shape):
    specs = [dict(id=k, origin=i, rate_bp=r, fx_pct=f) for k, (r, f) in specs_rf.items()]
    return forecast(pipe, mk, specs, L, variant, shape, batch_items=64, log=None)


def main():
    ap = argparse.ArgumentParser(description="Chronos-2 시나리오 계산 (BASE / STRESS)")
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--calib", default=None, help="calibrate.py 가 만든 보정 파일(json). 주면 5일 뒤 범위를 보정해서 보여줌")
    ap.add_argument("--calib-mode", choices=["global", "lower", "by_vix"], default="global")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--rate", type=float, default=0.0, help="미국 10년 금리 다음 5거래일 변화(bp)")
    ap.add_argument("--fx", type=float, default=0.0, help="원달러 다음 5거래일 변화(%%)")
    ap.add_argument("--shape", choices=["linear", "front", "back"], default="linear", help="충격이 5일에 걸쳐 들어오는 모양")
    ap.add_argument("--date", default=None, help="기준일(YYYY-MM-DD). 없으면 가장 최근 날짜")
    ap.add_argument("--curve", action="store_true", help="충격 크기별 반응 곡선도 계산")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    mk = Market(load_state())
    i = mk.pos(a.date) if a.date else mk.n - 1
    asof = mk.date(i)
    pipe = FakePipeline() if a.fake else load_pipeline(a.model, a.device)
    meta = None if a.fake else read_meta(a.model)

    res = run(pipe, mk, i, {"BASE": (0.0, 0.0), "STRESS": (a.rate, a.fx)}, a.context, a.variant, a.shape)
    qc = [str(q) for q in TAUS]
    names = {qc[0]: "5일 뒤 수익률 하단10%", qc[1]: "5일 뒤 수익률 중앙", qc[2]: "5일 뒤 수익률 상단90%"}
    s5 = res[res.step == H].set_index("id")
    table = pd.DataFrame({"BASE": s5.loc["BASE", qc], "STRESS": s5.loc["STRESS", qc]})
    if a.calib:                                           # 5일 뒤 범위(하단·상단)만 보정. 중앙값은 그대로.
        cal = load_calibration(a.calib)
        vix_now = float(mk.col["vix"][i])
        for k in ("BASE", "STRESS"):
            table[k] = apply_calibration(table[k].to_numpy()[None, :], [vix_now], cal, a.calib_mode)[0]
    table["차이(STRESS-BASE)"] = table["STRESS"] - table["BASE"]
    table.index = [names[c] for c in table.index]

    print(f"\n[기준일] {asof:%Y-%m-%d}   [입력] 모델={a.model}, 컨텍스트 {a.context}거래일, 입력 구성={a.variant}"
          + (f", 구간 보정={a.calib_mode}" if a.calib else ""))
    print(f"[충격] 미국 10년 금리 {a.rate:+g}bp, 원달러 {a.fx:+g}%  (5거래일, {a.shape} 경로; BASE 는 둘 다 0)")
    print("\n" + table.round(2).to_string())
    print("\n[1~5일차 분위수: 하단10% / 중앙 / 상단90%]")
    path = res.pivot(index="step", columns="id", values=qc)
    for k in ("BASE", "STRESS"):
        print("  " + k.ljust(8) + "  ".join(f"{h}일차 {path[(qc[0], k)][h]:+.2f}/{path[(qc[1], k)][h]:+.2f}/{path[(qc[2], k)][h]:+.2f}"
                                          for h in range(1, H + 1)))

    # 입력 범위 경고 (기준일까지 관측된 5일 변화의 1~99% 지점)
    warns = []
    for c, v, u in ((KNOWN[0], a.rate, "bp"), (KNOWN[1], a.fx, "%")):
        hist = mk.col[c][5:i + 1]
        lo, hi = np.quantile(hist, [.01, .99])
        if not lo <= v <= hi:
            warns.append(f"   - {c}: {v:+g}{u} (기준일까지 관측 범위 {lo:+.1f} ~ {hi:+.1f}{u})")
    if warns:
        print("\n⚠ 과거 5일 변화 분포(1~99%) 밖의 충격입니다. 결과를 신뢰하기 어렵습니다:\n" + "\n".join(warns))
    if meta and asof <= pd.Timestamp(meta["train_end"]):
        print(f"\n※ 기준일이 이 모델의 학습 구간({meta['train_start']}~{meta['train_end']}) 안이라 결과가 낙관적일 수 있습니다.")
    print("\n※ 5일 중 최저 수익률과 하락 확률은 이 모델에서 제공하지 않습니다. 과거 자료에서 배운 조건부 관계이며 인과효과나 미래 보장이 아닙니다.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = table.copy()
    out.insert(0, "항목", out.index)
    f = RESULTS / f"scenario_{asof:%Y-%m-%d}_rate{a.rate:+g}_fx{a.fx:+g}_{stamp}.csv"
    out.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"\n저장: results/{f.name}")

    if a.curve:
        rates = list(range(-50, 51, 10))
        fxs = [x / 2 for x in range(-10, 11, 2)]
        spec = {f"rate{r:+d}": (float(r), 0.0) for r in rates}
        spec.update({f"fx{x:+.1f}": (0.0, float(x)) for x in fxs})
        cr = run(pipe, mk, i, spec, a.context, a.variant, a.shape)
        c5 = cr[cr.step == H].set_index("id")[qc]
        c5.columns = ["q10", "q50", "q90"]
        c5.insert(0, "kind", ["rate_bp" if k.startswith("rate") else "fx_pct" for k in c5.index])
        c5.insert(1, "shock", [float(k[4:]) if k.startswith("rate") else float(k[2:]) for k in c5.index])
        cf = RESULTS / f"curve_{asof:%Y-%m-%d}_{stamp}.csv"
        c5.reset_index(drop=True).to_csv(cf, index=False, encoding="utf-8-sig")
        print("\n[반응 곡선] 같은 시장 상태에서 충격만 바꿨을 때 5일 뒤 수익률 (하단10% / 중앙 / 상단90%)")
        for kind, label in (("rate_bp", "미국 10년 금리(bp)"), ("fx_pct", "원달러(%)")):
            d = c5[c5.kind == kind].sort_values("shock")
            print(f"  {label}: " + " | ".join(f"{s:+g}: {a_:+.2f}/{b:+.2f}/{c:+.2f}" for s, a_, b, c in zip(d.shock, d.q10, d.q50, d.q90)))
            dm = np.diff(d.q50.to_numpy())
            if np.ptp(d.q50.to_numpy()) < 1e-9:
                print("    중앙값이 충격에 전혀 반응하지 않습니다 (이 입력 구성에서는 충격이 결과에 영향을 주지 않음)")
            else:
                print(f"    중앙값이 충격에 따라 단조 감소하는가: {'예' if (dm <= 1e-9).all() else '아니오'} "
                      f"(증가 구간 {int((dm > 1e-9).sum())}곳) — 환율은 대체로 감소해야 자연스럽고, 금리는 시대에 따라 방향이 달라 단조일 필요는 없습니다")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            fig, axes = plt.subplots(2, 1, figsize=(7, 7))
            for ax, (kind, label) in zip(axes, (("rate_bp", "미국 10년 금리 5일 변화(bp)"), ("fx_pct", "원달러 5일 변화(%)"))):
                d = c5[c5.kind == kind].sort_values("shock")
                ax.fill_between(d.shock, d.q10, d.q90, alpha=.2, color="#2a78d6")
                ax.plot(d.shock, d.q50, color="#2a78d6", lw=2)
                ax.axhline(0, color="#999", lw=.8)
                ax.set_xlabel(label)
                ax.set_ylabel("5일 뒤 코스피 수익률(%)")
                ax.set_title(f"{asof:%Y-%m-%d} 기준: 하단10% ~ 상단90% 범위와 중앙값", loc="left", fontsize=10)
            fig.tight_layout()
            pf = RESULTS / f"curve_{asof:%Y-%m-%d}_{stamp}.png"
            fig.savefig(pf, dpi=130)
            print(f"\n저장: results/{cf.name}, results/{pf.name}")
        except ImportError:
            print(f"\n저장: results/{cf.name} (matplotlib 가 없어 그림은 생략)")


if __name__ == "__main__":
    main()
