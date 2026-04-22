"""
超参数调优脚本：批量调用 train.py / evaluate.py 并汇总结果。

示例：
    python tune.py --lrs 1e-4,8e-5 --dropouts 0.3,0.25 --weight-decays 1e-4 --num-epochs 20 --patience 5
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

import config


def _parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _sanitize_tag(tag):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)


def run_cmd(cmd):
    print("执行:", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"命令失败，退出码={proc.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="谵妄识别超参数调优")
    parser.add_argument("--lrs", type=str, default="1e-4", help="学习率列表，逗号分隔")
    parser.add_argument("--weight-decays", type=str, default="1e-4", help="权重衰减列表，逗号分隔")
    parser.add_argument("--dropouts", type=str, default="0.3", help="dropout列表，逗号分隔")
    parser.add_argument("--batch-sizes", type=str, default="8", help="batch size列表，逗号分隔")
    parser.add_argument("--num-epochs", type=int, default=config.NUM_EPOCHS, help="训练轮数")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="早停耐心值")
    parser.add_argument("--max-frames", type=int, default=config.MAX_FRAMES_PER_CLIP, help="每片段最大帧数")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="随机种子")
    parser.add_argument("--max-trials", type=int, default=0, help="最多执行多少组（0=全部）")
    parser.add_argument("--prefix", type=str, default="tune", help="实验tag前缀")
    parser.add_argument("--use-audio", type=str, default="true", help="是否使用音频分支 true/false")
    args = parser.parse_args()

    lrs = _parse_float_list(args.lrs)
    wds = _parse_float_list(args.weight_decays)
    drops = _parse_float_list(args.dropouts)
    bss = [int(float(x.strip())) for x in args.batch_sizes.split(",") if x.strip()]

    combos = list(itertools.product(lrs, wds, drops, bss))
    if args.max_trials > 0:
        combos = combos[:args.max_trials]

    if not combos:
        print("没有可执行的参数组合")
        return

    py = sys.executable
    summary = []

    for i, (lr, wd, dr, bs) in enumerate(combos, start=1):
        raw_tag = f"{args.prefix}_{i}_lr{lr}_wd{wd}_dr{dr}_bs{bs}"
        tag = _sanitize_tag(raw_tag)
        print("\n" + "=" * 70)
        print(f"Trial {i}/{len(combos)} | tag={tag}")

        train_cmd = [
            py, "train.py",
            "--tag", tag,
            "--learning-rate", str(lr),
            "--weight-decay", str(wd),
            "--dropout", str(dr),
            "--batch-size", str(bs),
            "--num-epochs", str(args.num_epochs),
            "--patience", str(args.patience),
            "--max-frames", str(args.max_frames),
            "--seed", str(args.seed),
            "--use-audio", args.use_audio,
        ]
        run_cmd(train_cmd)

        eval_cmd = [py, "evaluate.py", "--tag", tag]
        run_cmd(eval_cmd)

        eval_path = os.path.join(config.LOG_DIR, f"evaluation_results_{tag}.json")
        test_path = os.path.join(config.LOG_DIR, f"test_results_{tag}.json")

        row = {
            "tag": tag,
            "learning_rate": lr,
            "weight_decay": wd,
            "dropout": dr,
            "batch_size": bs,
            "test_auc": None,
            "test_accuracy": None,
            "youden_auc": None,
            "youden_accuracy": None,
            "youden_f1": None,
            "youden_threshold": None,
        }

        if os.path.isfile(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                tj = json.load(f)
            row["test_auc"] = tj.get("test_auc")
            row["test_accuracy"] = tj.get("accuracy")

        if os.path.isfile(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                ej = json.load(f)
            row["youden_auc"] = ej.get("auc")
            row["youden_accuracy"] = ej.get("accuracy")
            row["youden_f1"] = ej.get("f1")
            row["youden_threshold"] = ej.get("optimal_threshold")

        summary.append(row)

    summary.sort(key=lambda x: x.get("youden_f1") or -1, reverse=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = os.path.join(config.LOG_DIR, f"tuning_summary_{ts}.json")
    summary_csv = os.path.join(config.LOG_DIR, f"tuning_summary_{ts}.csv")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n调参完成")
    print(f"汇总JSON: {summary_json}")
    print(f"汇总CSV:  {summary_csv}")
    if summary:
        best = summary[0]
        print("最佳组合:")
        print(best)


if __name__ == "__main__":
    main()
