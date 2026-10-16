# -*- coding: utf-8 -*-
"""
파인튜닝 설정(학습률 × 스텝) 여러 개를 차례로 학습 → valid 채점해서 한 표로 비교한다. 설정은 valid 로만 고른다.

  각 설정마다 finetune.py(--upto 2018) → evaluate.py(--split valid) 를 실행하고, 마지막에 compare.py 로 전부 비교한다.
  오래 걸린다 (설정 수 × 학습 시간). 먼저 --dry 로 실행할 명령만 확인하세요.

실행 예
  python src/sweep.py --variant full --dry
  python src/sweep.py --variant full --lrs 1e-5,5e-5 --steps 200,400
  python src/sweep.py --variant full --base autogluon/chronos-2-small       # CPU 일 때
"""
import argparse
import subprocess
import sys
from itertools import product

from common import ROOT, VARIANTS


def main():
    ap = argparse.ArgumentParser(description="파인튜닝 설정 비교 (학습률 × 스텝)")
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--lrs", default="1e-5,5e-5", help="학습률 후보 (쉼표로)")
    ap.add_argument("--steps", default="200,400", help="스텝 후보 (쉼표로, 100의 배수 권장)")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--val-years", type=int, default=2)
    ap.add_argument("--base", default="amazon/chronos-2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-items", type=int, default=32, help="평가할 때 한 번에 예측할 기준일 수 (메모리 부족하면 줄이기)")
    ap.add_argument("--dry", action="store_true", help="명령만 출력하고 실행하지 않음")
    a = ap.parse_args()

    py = sys.executable
    cmds = []
    for lr, st in product([x.strip() for x in a.lrs.split(",")], [x.strip() for x in a.steps.split(",")]):
        tag = f"sw_{a.variant}_lr{lr}_s{st}"
        cmds.append([py, "src/finetune.py", "--upto", "2018", "--variant", a.variant, "--lr", lr, "--steps", st,
                     "--context", str(a.context), "--val-years", str(a.val_years), "--base", a.base,
                     "--device", a.device, "--tag", tag])
        cmds.append([py, "src/evaluate.py", "--split", "valid", "--model", f"models/{tag}/finetuned-ckpt",
                     "--variants", a.variant, "--context", str(a.context), "--device", a.device,
                     "--batch-items", str(a.batch_items)])
    cmds.append([py, "src/compare.py", "--split", "valid"])

    print(f"실행할 명령 {len(cmds)}개 (설정 {(len(cmds) - 1) // 2}개 × [학습 + 채점] + 비교)")
    for c in cmds:
        print("  " + " ".join(c[1:]))
    if a.dry:
        return
    for c in cmds:
        print("\n▶ " + " ".join(c[1:]))
        subprocess.run(c, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
