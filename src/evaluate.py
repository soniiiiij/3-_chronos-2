# -*- coding: utf-8 -*-
"""
Chronos-2 평가: 제로샷(amazon/chronos-2) 또는 파인튜닝한 체크포인트를 kNN 과 같은 행·정답·지표로 채점한다.

  - 행·정답   : model_training/data/train_table.csv 의 valid(2019~2022) 또는 test(2023~) 행, 정답 = 5일 뒤 수익률
  - 입력      : 기준일 t 까지의 최근 L거래일(시장 상태) + t 이후 5일의 충격 경로 (정답을 아는 것은 '충격 총량 2개'뿐)
  - 기준선    : ① 과거 분포 ② 변동성 스케일 (kNN 6단계와 같은 방식)  /  test 에서는 기존 kNN 결과도 나란히 표시
  - 채점 지표 : pinball, 과거 분포 대비 개선(skill)과 95% 구간, 80% 포함률, 하단/상단 이탈률, 범위 폭

  ※ valid 로 입력 구성·컨텍스트 길이·파인튜닝 설정을 고르고, test 는 마지막에 한 번만(--final) 채점한다.
     (기존 kNN 작업과 같은 원칙. test 결과를 보고 다시 고르지 않는다.)

실행 예
  python src/evaluate.py --split valid --model amazon/chronos-2 --variants full,shock_only,no_cov --context 512
  python src/evaluate.py --split valid --model models/lora_upto2018/finetuned-ckpt --variants full --context 512
  python src/evaluate.py --split test --final --model models/lora_upto2022/finetuned-ckpt --variants full --context 512
  python src/evaluate.py --split valid --fake --limit 100        # chronos 없이 배관만 점검 (성능 비교 아님)
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd

from common import (H, TAUS, SHOCK_COLS, Y_RET, KNN_TEST_PRED, RESULTS, VARIANTS, Market, load_state, load_truth,
                    forecast, FakePipeline, load_pipeline, read_meta, baselines, score,
                    apply_calibration, load_calibration)


def run_info(pipe, a, rows, variants) -> dict:
    """감사용 실행 기록: 모델 체크포인트·라이브러리 버전·입력 설정. 알 수 없는 값은 '미확인'으로 남긴다."""
    def ver(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return "미확인"
    rev = "미확인"
    try:                                           # 허브에서 내려받은 모델의 revision(커밋 해시). 로컬 폴더면 없을 수 있음
        rev = getattr(pipe.model.config, "_commit_hash", None) or "미확인"
    except Exception:
        pass
    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": a.model, "model_revision(commit)": rev, "fake_pipeline": bool(a.fake),
        "chronos-forecasting": ver("chronos-forecasting"), "torch": ver("torch"), "transformers": ver("transformers"),
        "pandas": pd.__version__, "numpy": np.__version__, "python": sys.version.split()[0], "platform": platform.platform(),
        "split": a.split, "n_rows": int(len(rows)), "first_origin": str(rows["date"].min().date()),
        "last_origin": str(rows["date"].max().date()), "variants": variants, "context": str(a.context),
        "shock_path": "zero(진단)" if a.zero_shock else a.path, "shock_shape": a.shape, "device": a.device,
        "batch_items": a.batch_items, "limit": a.limit,
        "shock_note": "충격값 = 해당 5거래일에 실제로 일어난 누적 변화(학습표 shock 열). 정답을 아는 조건부 진단이며 사전 예측이 아니다."
                      if not a.zero_shock else "충격 0 고정(진단)",
        "outputs": [],
    }


def main():
    ap = argparse.ArgumentParser(description="Chronos-2 평가 (kNN 과 같은 행·정답·지표)")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--final", action="store_true", help="test 를 채점하겠다는 확인 (마지막에 한 번만)")
    ap.add_argument("--model", default="amazon/chronos-2", help="허브 이름 또는 파인튜닝 체크포인트 폴더")
    ap.add_argument("--variants", default="full",
                    help="입력 구성을 쉼표로: full(기본) / all(기존 입력 전부) / select(직접 고른 열) / shock_only / no_cov / "
                         "past_only(B) / rate_only(C) / fx_only(D)")
    ap.add_argument("--context", default="512", help="컨텍스트 길이(거래일). 쉼표로 여러 개 가능: 256,512,1024")
    ap.add_argument("--path", choices=["ramp", "realized"], default="ramp",
                    help="ramp: 충격 총량만 알려주고 경로는 나눠서(앱과 같은 조건, 기본) / realized: 실제 일별 경로")
    ap.add_argument("--shape", choices=["linear", "front", "back"], default="linear", help="ramp 일 때 충격이 들어오는 모양")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-items", type=int, default=32, help="한 번에 예측할 기준일 수 (메모리 부족하면 줄이기)")
    ap.add_argument("--tag", default=None, help="결과 파일 이름표 (기본: 모델 이름)")
    ap.add_argument("--limit", type=int, default=None, help="앞쪽 N일만 (빠른 점검용. 성능 비교에는 쓰지 말 것)")
    ap.add_argument("--fake", action="store_true", help="모의 예측기로 배관만 점검")
    ap.add_argument("--zero-shock", action="store_true",
                    help="실제 충격 대신 모든 날 (0bp, 0%)를 넣는다. '충격 없음=0'과 '충격 열 자체를 뺌'(past_only)을 구분하는 진단용. "
                         "--tag 를 꼭 따로 주세요(예: --tag zero).")
    ap.add_argument("--calib", default=None,
                    help="calibrate.py 가 만든 보정 파일(json). 주면 보정한 결과를 '+보정' 행으로 함께 채점")
    ap.add_argument("--calib-mode", choices=["global", "lower", "by_vix"], default="global",
                    help="보정 방식 (전체 배율 / 하단만 / VIX 구간별)")
    a = ap.parse_args()

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"알 수 없는 입력 구성 {bad}. 사용 가능: {', '.join(VARIANTS)}")
    cal = load_calibration(a.calib) if a.calib else None

    if a.split == "test" and not a.final and not a.fake:
        raise SystemExit("test(2023~)는 모델을 다 정한 뒤 마지막에 한 번만 채점합니다. 확인했다면 --final 을 붙이세요.")
    RESULTS.mkdir(exist_ok=True)

    mk, truth = Market(load_state()), load_truth()
    rows = truth[truth.split == a.split].reset_index(drop=True)
    if a.limit:
        rows = rows.iloc[:a.limit]
    dev = truth[truth.split == "train"] if a.split == "valid" else truth[truth.split != "test"]
    first = rows["date"].iloc[0]

    # 누수 방지: 파인튜닝이 평가 구간 이후 자료까지 봤다면 중단
    meta = None if a.fake else read_meta(a.model)
    if meta:
        te = pd.Timestamp(meta["train_end"])
        if te > first:
            raise SystemExit(f"누수 위험: 이 모델은 {te:%Y-%m-%d} 까지 학습했는데 평가 첫날은 {first:%Y-%m-%d} 입니다. "
                             f"{a.split} 평가에는 그보다 앞서 끝난 모델을 쓰세요.")
        print(f"파인튜닝 모델: {meta['mode']}, {meta['train_start']} ~ {meta['train_end']} 학습, {meta['steps']}스텝")
    tag = a.tag or (("fake" if a.fake else (a.model.replace("/", "_") if not meta else meta.get("tag", "ft"))))

    if a.zero_shock and a.path == "realized":
        raise SystemExit("--zero-shock 은 --path ramp 에서만 씁니다.")
    if a.zero_shock and not a.tag:
        raise SystemExit("--zero-shock 은 결과 파일이 덮어쓰이지 않게 --tag 를 직접 주세요 (예: --tag zero).")
    # ※ 평가의 충격값 = 그 5거래일 동안 '실제로 일어난' 누적 변화(학습표 shock 열). 정답을 알고 넣는 조건부 진단이며 사전 예측이 아니다.
    specs = [dict(id=str(d.date()), origin=mk.pos(d),
                  rate_bp=0.0 if a.zero_shock else float(r), fx_pct=0.0 if a.zero_shock else float(f),
                  realized=(a.path == "realized"))
             for d, r, f in zip(rows["date"], rows[SHOCK_COLS[0]], rows[SHOCK_COLS[1]])]
    ids = [s["id"] for s in specs]
    print(f"\n[{a.split}] {rows['date'].min():%Y-%m-%d} ~ {rows['date'].max():%Y-%m-%d}  {len(rows):,}일  "
          f"충격 경로={'0 고정(진단)' if a.zero_shock else a.path}{'' if a.path == 'realized' else '/' + a.shape}")

    pipe = FakePipeline() if a.fake else load_pipeline(a.model, a.device)
    info = run_info(pipe, a, rows, variants)
    models = baselines(dev, rows)
    for variant in variants:
        for L in [int(x) for x in str(a.context).split(",")]:
            name = f"chronos2_{variant}_L{L}"
            t0 = time.time()
            print(f"  {name} 예측 중 ...")
            res = forecast(pipe, mk, specs, L, variant, a.shape, a.batch_items)
            s5 = res[res.step == H].set_index("id").loc[ids]
            q = s5[[str(x) for x in TAUS]].to_numpy()
            models[name] = q
            if cal is not None:
                models[name + "+보정"] = apply_calibration(q, rows["vix"].to_numpy(), cal, a.calib_mode)
            out = pd.DataFrame({"date": rows["date"].values, "y_ret": rows[Y_RET].values,
                                "pred_ret_q10": q[:, 0], "pred_ret_q50": q[:, 1], "pred_ret_q90": q[:, 2],
                                "vix": rows["vix"].values, SHOCK_COLS[0]: rows[SHOCK_COLS[0]].values,
                                SHOCK_COLS[1]: rows[SHOCK_COLS[1]].values})
            f = RESULTS / f"eval_{a.split}_{tag}_{variant}_L{L}_predictions.csv"
            out.to_csv(f, index=False, encoding="utf-8-sig")
            print(f"    {time.time() - t0:.0f}초, 저장: results/{f.name}")
            info["outputs"].append({"variant": variant, "context": L, "predictions_file": f.name, "n_rows": int(len(out)),
                                    "seconds": round(time.time() - t0)})
            (RESULTS / f"eval_{a.split}_{tag}_{variant}_L{L}_runinfo.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    if a.split == "test" and KNN_TEST_PRED.exists() and not a.limit:
        k = pd.read_csv(KNN_TEST_PRED, parse_dates=["date"]).set_index("date").loc[rows["date"]]
        models["기존_kNN(참고)"] = k[["pred_ret_q10", "pred_ret_q50", "pred_ret_q90"]].to_numpy()

    vix_hi = truth.loc[truth.split == "train", "vix"].quantile(2 / 3)
    M = score(models, rows[Y_RET], rows["vix"], vix_hi)
    fm = RESULTS / f"eval_{a.split}_{tag}_{'-'.join(variants)}_metrics.csv"      # 구성별로 따로 저장 (서로 덮어쓰지 않음)
    M.to_csv(fm, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\n=== {a.split} 성적 (5일 뒤 수익률 범위, 목표: 80% 포함률 0.80 / 하단·상단 이탈 각 0.10) ===")
    print(M.round(3).to_string(index=False))
    print(f"\n저장: results/{fm.name}")
    if cal is not None and a.split == "valid":
        print("※ '+보정' 행은 보정 배율을 valid 자료로 만들었다면 낙관적인 점수입니다. valid 에서의 정직한 확인은 "
              "calibrate.py 의 표(앞 기간으로 보정 → 뒤 기간에서 확인)를 보세요.")
    if a.fake:
        print("※ 모의 예측기 결과입니다. 성능을 뜻하지 않습니다.")
    if a.limit:
        print("※ --limit 로 일부만 채점했습니다. 성능 비교에는 전체 구간을 쓰세요.")


if __name__ == "__main__":
    main()
