# -*- coding: utf-8 -*-
"""Fill the {{PLACEHOLDER}} slots in results/rollout_v4/conclusion.md from the
merged 19-entry metrics_unified.json (test-side numbers).

Usage: python fill_conclusion_v4.py [EVAL_JSON] [CONCLUSION_MD]
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/rollout_v4/eval_unified/metrics_unified.json"
MD = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/rollout_v4/conclusion.md"

d = json.loads(JSON.read_text(encoding="utf-8"))
E = d["entries"]


def f1(entry, split, h):
    m = E[entry]["rollout"][split]
    key = str(h)
    if key not in m:
        return None
    v = m[key].get("spike_f1")
    return None if v is None else float(v)


def eh(entry, split, key):
    v = E[entry]["effective_horizon"][split].get(key)
    return "none" if v is None else v


def silent(entry, split):
    return eh(entry, split, "silent_collapse_step")


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


rep = {}

# --- {{S5_COLLAPSE}} / {{S5_F1_10}} -------------------------------------
rep["S5_COLLAPSE"] = str(silent("S5", "test_seen"))
rep["S5_F1_10"] = fmt(f1("S5", "test_seen", 10))

# --- {{K_ABLATION_TABLE}} ------------------------------------------------
rows = []
for lab, k in (("p1temp", 32), ("p1temp_k16", 16), ("p1temp_k1", 1)):
    rows.append(
        f"| k={k:<2d} | {fmt(f1(lab,'test_seen',10))} | {fmt(f1(lab,'test_seen',20))} "
        f"| {fmt(f1(lab,'test_seen',50))} | {fmt(f1(lab,'test_seen',100))} "
        f"| {eh(lab,'test_seen','H_F1_0.9')} | {silent(lab,'test_seen')} "
        f"| {fmt(f1(lab,'test_ood',20))} |")
rep["K_ABLATION_TABLE"] = (
    "\n| 历史长度 | F1@10 seen | F1@20 seen | F1@50 seen | F1@100 seen "
    "| H_F1_0.9 seen | 静默崩溃步 seen | F1@20 ood |\n"
    "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
)
f10_32, f10_16, f10_1 = (f1(l, "test_seen", 10) for l in ("p1temp", "p1temp_k16", "p1temp_k1"))
f20_32, f20_16, f20_1 = (f1(l, "test_seen", 20) for l in ("p1temp", "p1temp_k16", "p1temp_k1"))
if f10_32 >= f10_16 >= f10_1 and (f10_32 - f10_1) > 0.02:
    verdict = (f"单调：k32({f10_32:.3f}) ≥ k16({f10_16:.3f}) > k1({f10_1:.3f})，"
               f"同权重下历史长度带来 F1@10 +{(f10_32-f10_1):.3f} 的增益 —— "
               "历史信息有真实价值，Markov 假设确实损失信息")
elif abs(f10_32 - f10_1) <= 0.02:
    verdict = (f"k32({f10_32:.3f}) ≈ k1({f10_1:.3f})，差异 ≤0.02 —— "
               "在已经训好的时序权重上，历史截断几乎无影响"
               "（但 k1 覆盖破坏位置编码的原生训练结果见 §8，两者合读："
               "历史的价值在训练期已被吸收进权重，评估期截断影响小）")
else:
    verdict = (f"非单调：k32={f10_32:.3f}, k16={f10_16:.3f}, k1={f10_1:.3f} —— "
               "历史长度的收益不稳定")
rep["K_ABLATION_VERDICT"] = "\n\n判定：" + verdict

# --- {{T2_VS_P1TEMP}} -----------------------------------------------------
t2 = [fmt(f1("T2", "test_seen", h)) for h in (10, 20, 50, 100)]
pt = [fmt(f1("p1temp", "test_seen", h)) for h in (10, 20, 50, 100)]
rep["T2_VS_P1TEMP"] = (
    f"\n\nT2（k16 微调）vs p1temp（k32 零训练）test_seen F1@10/20/50/100："
    f"{' / '.join(t2)} vs {' / '.join(pt)}。"
    + ("微调在该架构上仍为净负。" if (f1('T2','test_seen',10) or 0) < (f1('p1temp','test_seen',10) or 0)
       else "微调在该对比上非负。")
)

# --- {{TEST_TABLE_1/2/3}} from merged summary_tables.md -------------------
st = (JSON.parent / "summary_tables.md").read_text(encoding="utf-8")
blocks = re.split(r"^## ", st, flags=re.M)
def grab(n):
    for b in blocks:
        if b.startswith(f"Table {n} "):
            lines = b.splitlines()
            return "\n".join(lines[1:]).strip("\n")  # drop the heading line
    raise SystemExit(f"Table {n} not found in summary_tables.md")
for n in (1, 2, 3):
    rep[f"TEST_TABLE_{n}"] = "\n" + grab(n) + "\n"

# --- {{OOD_VERDICT}} -------------------------------------------------------
drops = []
for lab in E:
    s = f1(lab, "test_seen", 100)
    o = f1(lab, "test_ood", 100)
    if s and o is not None:
        drops.append((lab, s, o, (s - o) / s))
drops.sort(key=lambda t: -t[3])
worst = drops[0]
rank_seen = sorted((l for l in E), key=lambda l: -(f1(l, "test_seen", 100) or 0))
rank_ood = sorted((l for l in E), key=lambda l: -(f1(l, "test_ood", 100) or 0))
top3_preserved = set(rank_seen[:3]) == set(rank_ood[:3])
rep["OOD_VERDICT"] = (
    f"test_ood vs test_seen @F1_100：最差退化 {worst[0]} "
    f"{worst[1]:.3f}→{worst[2]:.3f}（−{worst[3]*100:.0f}%）；"
    f"全场最大相对退化 −{max(d[3] for d in drops)*100:.0f}%，"
    f"前三排名 seen↔ood {'保持一致' if top3_preserved else '不一致'}。"
    + ("OOD 无悬崖：分布偏移平滑降级，无突发失效。"
       if worst[3] < 0.5 and top3_preserved else
       "存在明显 OOD 退化，需注意泛化边界。")
)

# --- {{HORIZON_TABLE}} -----------------------------------------------------
hdr = ("| 单元 | H_F1_0.9 s/ood | H_F1_0.7 s/ood | H_rate_0.8 s/ood "
       "| 静默崩溃 s/ood |\n|---|---|---|---|\n")
lines = []
order = [l for l in ("phase1", "A0", "A1", "A2", "A3", "S1", "S2", "S3", "S4",
                     "S5", "S6", "S7", "T2", "T3", "G1",
                     "p1temp", "p1temp_k16", "p1temp_k1", "p1wide") if l in E]
for lab in order:
    s, o = "test_seen", "test_ood"
    lines.append(
        f"| {lab} | {eh(lab,s,'H_F1_0.9')}/{eh(lab,o,'H_F1_0.9')} "
        f"| {eh(lab,s,'H_F1_0.7')}/{eh(lab,o,'H_F1_0.7')} "
        f"| {eh(lab,s,'H_rate_0.8')}/{eh(lab,o,'H_rate_0.8')} "
        f"| {silent(lab,s)}/{silent(lab,o)} |")
rep["HORIZON_TABLE"] = "\n" + hdr + "\n".join(lines) + "\n"

# --- apply ---------------------------------------------------------------
text = MD.read_text(encoding="utf-8")
missing = [k for k in rep if "{{" + k + "}}" not in text]
if missing:
    print(f"[warn] placeholders not found (already filled?): {missing}")
for k, v in rep.items():
    text = text.replace("{{" + k + "}}", v)
MD.write_text(text, encoding="utf-8")
left = re.findall(r"\{\{[A-Z0-9_]+\}\}", text)
print(f"[done] remaining placeholders: {left if left else 'none'}")
