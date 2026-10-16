# -*- coding: utf-8 -*-
"""
Chronos-2 파인튜닝 (구성 3: 더 나은 학습법). 제로샷 결과를 본 뒤, 우리 자료로 모델을 조금 더 맞출 때 쓴다.

  학습 자료 : 2001-03-02(kNN 학습 시작일과 같음) ~ --upto 년도 마지막 거래일의 연속 일별 시계열 1개
              (타깃 = 코스피 후행 5일 수익률, 미래 공변량 = 금리·환율 후행 5일 변화, 과거 공변량 = --variant 에 따라 다름)
              Chronos-2 가 이 시계열에서 '과거 구간 → 다음 5일' 조각을 무작위로 잘라 학습한다. (행 단위 표가 아님)
  --upto 2018 : valid(2019~2022) 평가용.   --upto 2022 : 최종 모델(test 2023~ 평가용)

  그냥 학습하는 것보다 나은 점 (구성 3의 학습 방식)
    1) 검증으로 학습 시점 고르기 (--val-years 2, 기본)
         자료의 마지막 2년은 학습에서 빼 두고, 100스텝마다 그 구간의 예측 오차를 재서 가장 좋았던 시점의 모델을 저장한다.
         (너무 오래 학습해서 과거를 외워 버리는 것을 막는다. 그 2년은 5거래일마다 '기준일 → 5일 뒤' 시험 문제 하나씩 쓴다.)
    2) LoRA (기본): 모델 전체가 아니라 작은 보조 가중치만 학습한다.
         서로 겹치는 5일 정답이 많아 실질 표본이 약 1,200개뿐이라 전체 학습은 과적합하기 쉽다.
    3) 배치 크기 자동 계산: 열(공변량)이 많을수록 한 배치에 들어가는 '기준일' 수가 줄어든다. 기준일 N개(--windows-per-batch)를 기준으로 맞춘다.
    4) 누수 방지: 평가 구간 자료는 학습에 들어가지 않는다. 체크포인트에 train_meta.json 을 남겨 evaluate.py 가 누수를 검사한다.
  이후 구간 보정(calibrate.py)으로 '80% 범위가 실제로 80% 가 되도록' 한 번 더 맞춘다.

실행 예
  python src/finetune.py --variant full --tag m3_full                            # 구성 3 기본 (제로샷 valid 에서 full 이 가장 좋았음)
  python src/finetune.py --variant full --lr 5e-5 --steps 400 --tag m3_full_lr5
  python src/finetune.py --upto 2022 --variant full --tag final_full             # valid 로 설정을 정한 뒤 최종용
  (CPU 만 있다면) python src/finetune.py --base autogluon/chronos-2-small --variant full --tag m3_small
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from common import H, TARGET, VARIANTS, MODELS, load_state, load_truth


def make_items(ser: pd.DataFrame, variant: str, n_fit: int, context: int):
    """학습용 시계열 1개(앞 n_fit일)와 검증용 시계열 여러 개를 dict 로 만든다 (torch 없이도 실행됨).

    검증용: n_fit 이후 5거래일마다 하나. 각 시계열의 마지막 H(5)개가 '정답 구간'이고, 그 앞 context 개가 입력이다.
    정답 구간 마지막 값 = 기준일의 5일 뒤 수익률 (타깃이 후행 5일 수익률이라서)."""
    past_cols, known_cols = VARIANTS[variant]
    cols = list(past_cols) + list(known_cols)

    def one(seg: pd.DataFrame) -> dict:
        d = {"target": seg[TARGET].to_numpy(np.float32)}
        if cols:
            d["past_covariates"] = {c: seg[c].to_numpy(np.float32) for c in cols}     # 미래 공변량의 과거 값도 여기에 둔다
        return d

    train = one(ser.iloc[:n_fit])
    val = []
    for e in range(n_fit + H - 1, len(ser), H):
        lo = max(0, e + 1 - (context + H))
        val.append(one(ser.iloc[lo:e + 1]))
    return train, val


def main():
    ap = argparse.ArgumentParser(description="Chronos-2 파인튜닝 (검증으로 학습 시점을 고르는 LoRA)")
    ap.add_argument("--upto", type=int, choices=[2018, 2022], default=2018, help="이 연도 마지막 거래일까지의 자료 사용")
    ap.add_argument("--variant", choices=list(VARIANTS), default="full", help="입력 구성 (full / select / all ...)")
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--steps", type=int, default=400, help="학습 스텝 (검증은 100스텝마다)")
    ap.add_argument("--lr", type=float, default=1e-5, help="LoRA 는 1e-5~1e-4, full 은 1e-6~1e-5 권장")
    ap.add_argument("--val-years", type=int, default=2, help="자료 끝의 N년을 검증용으로 빼 둠 (0 이면 검증 없이 마지막 스텝 사용)")
    ap.add_argument("--windows-per-batch", type=int, default=8, help="한 배치에 넣을 '기준일' 수 (배치 크기 = 이 값 × 열 수)")
    ap.add_argument("--batch-size", type=int, default=None, help="직접 지정하면 위 계산을 덮어씀 (열 수를 포함한 시계열 개수)")
    ap.add_argument("--context", type=int, default=512, help="학습·예측에 쓸 컨텍스트 길이(거래일)")
    ap.add_argument("--min-past", type=int, default=256, help="학습 조각이 최소한 가져야 하는 과거 길이")
    ap.add_argument("--base", default="amazon/chronos-2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"{a.mode}_{a.variant}_upto{a.upto}"

    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.preprocess import from_list_of_dicts
    if a.mode == "lora":
        try:
            import peft  # noqa: F401
        except ImportError:
            raise SystemExit("LoRA 파인튜닝에는 peft 가 필요합니다:  pip install peft   (없으면 chronos 가 조용히 full 모드로 바뀝니다)")

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    # 자료: kNN 과 같은 시작일 ~ upto 년 마지막 거래일. 끝의 val-years 년은 검증용으로 분리
    state, truth = load_state(), load_truth()
    start = truth["date"].min()
    data_end = state.loc[state["date"].dt.year <= a.upto, "date"].max()
    ser = state[(state["date"] >= start) & (state["date"] <= data_end)].reset_index(drop=True)
    if a.val_years > 0:
        fit_end = state.loc[state["date"].dt.year <= a.upto - a.val_years, "date"].max()
        n_fit = int((ser["date"] <= fit_end).sum())
    else:
        n_fit = len(ser)
    past_cols, known_cols = VARIANTS[a.variant]
    n_vars = 1 + len(past_cols) + len(known_cols)
    batch_size = a.batch_size or max(8, a.windows_per_batch * n_vars)

    train_item, val_items = make_items(ser, a.variant, n_fit, a.context)
    kn = list(known_cols) or None
    inputs = from_list_of_dicts([train_item], prediction_length=H, known_covariates_names=kn)
    val_inputs = from_list_of_dicts(val_items, prediction_length=H, known_covariates_names=kn) if val_items else None

    fit_last = ser["date"].iloc[n_fit - 1]
    print(f"입력 구성={a.variant} (과거 공변량 {len(past_cols)}, 미래 공변량 {len(known_cols)}, 합계 {n_vars}열)")
    print(f"학습 시계열: {start:%Y-%m-%d} ~ {fit_last:%Y-%m-%d}  {n_fit:,}거래일")
    if val_items:
        print(f"검증: {ser['date'].iloc[n_fit]:%Y-%m-%d} ~ {data_end:%Y-%m-%d}, 5거래일마다 하나씩 {len(val_items)}개 "
              f"(100스텝마다 평가, 가장 좋은 시점의 모델을 저장)")
    else:
        print("검증 없이 마지막 스텝의 모델을 저장합니다 (--val-years 2 권장).")
    print(f"배치 크기 {batch_size} (= 기준일 약 {batch_size // n_vars}개 × {n_vars}열), {a.steps}스텝, 학습률 {a.lr:g}, {a.mode}")

    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("※ CPU 로 파인튜닝합니다. 오래 걸릴 수 있어요. --base autogluon/chronos-2-small 또는 --steps 줄이기를 고려하세요.")
    pipe = Chronos2Pipeline.from_pretrained(a.base, device_map=device)

    out_dir = MODELS / tag
    fit_kw = dict(prediction_length=H, finetune_mode=a.mode, learning_rate=a.lr, num_steps=a.steps,
                  batch_size=batch_size, context_length=a.context, min_past=a.min_past,
                  output_dir=str(out_dir), logging_steps=50)
    if val_inputs is not None:
        fit_kw["validation_inputs"] = val_inputs
    pipe.fit(inputs, **fit_kw)

    ckpt = out_dir / "finetuned-ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    meta = dict(tag=tag, base=a.base, mode=a.mode, variant=a.variant, steps=a.steps, lr=a.lr, batch_size=batch_size,
                windows_per_batch=a.windows_per_batch, context=a.context, min_past=a.min_past, seed=a.seed,
                val_years=a.val_years, n_val_windows=len(val_items),
                train_start=str(start.date()), train_fit_end=str(fit_last.date()),
                train_end=str(data_end.date()),                       # 검증용 구간까지 포함한 자료의 끝 (누수 검사 기준)
                n_days=int(len(ser)), created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    (ckpt / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {ckpt}")
    split = "valid" if a.upto == 2018 else "test --final"
    print(f"다음 단계:\n  python src/evaluate.py --split {split} --model models/{tag}/finetuned-ckpt "
          f"--variants {a.variant} --context {a.context}")


if __name__ == "__main__":
    main()
