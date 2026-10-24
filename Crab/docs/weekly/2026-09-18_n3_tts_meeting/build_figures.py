"""Read-only evidence audit; write aggregate meeting tables/figures in this folder.

No training, model loading, W&B calls, source-audio changes, or transcript export.
Run with Crab/.venv/bin/python; use OPENBLAS_NUM_THREADS=1.
"""
import ast
import csv
import hashlib
import json
import re
import statistics
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ASSETS = OUT / "assets"
TABLES = OUT / "tables"
SOURCES = {}
CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
E, U, N, DARK, GREY = "#D88722", "#8357A5", "#2B8594", "#18344B", "#65758B"
COLORS = [E, U, N]


def record(path):
    path = Path(path)
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    SOURCES[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def table(name, headers, rows):
    def esc(v):
        s = str(v)
        for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
            s = s.replace(a, b)
        return s
    cols = "l" * len(headers)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join(map(esc, headers)) + r"\\", r"\midrule"]
    lines += [" & ".join(map(esc, row)) + r"\\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (TABLES / f"{name}.tex").write_text("\n".join(lines))
    (TABLES / f"{name}.md").write_text(
        "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows) + "\n")


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(ASSETS / f"{name}.{ext}", bbox_inches="tight", dpi=170)
    plt.close(fig)


def panel(ax, x, y, w, h, title, body="", color=DARK, fs=12, title_fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        edgecolor=color, facecolor=color + "0C", linewidth=1.5))
    ax.text(x + .18, y + h - .22, title, color=color, weight="bold",
            fontsize=title_fs, va="top")
    if body:
        ax.text(x + .18, y + h - .70, body, color=DARK, fontsize=fs,
                va="top", linespacing=1.5)


def arrow(ax, a, b, color=GREY):
    ax.annotate("", xy=b, xytext=a,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=15))


def graph_axes(w=15, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, w), ylim=(0, h)); ax.axis("off")
    return fig, ax


def audit_data():
    script = record(ROOT / "scripts/build_nnime_crab_csv.py")
    sets = {}
    for node in ast.parse(script.read_text()).body:
        if isinstance(node, ast.Assign):
            key = getattr(node.targets[0], "id", "")
            if key in ["EXCITED", "UNCONFIDENT", "NEUTRAL", "DROP_ANY"]:
                sets[key] = ast.literal_eval(node.value)
    assert [len(sets[k]) for k in sets] == [9, 31, 6, 16]
    root = ROOT.parent / "datasets/NNIME"
    xlsx = root / ("Emotion Annotation of Sentence-Level by Native Rater/"
        "Emotion Annotation of Sentence-Level by Native Rater/"
        "Temporary_Version_Updated_20210601/Mandarin_Chinese_Version/Emotion.xlsx")
    raw = pd.read_excel(record(xlsx))
    speech = raw[raw.Type == "Speech"]
    drops, ties, unknown = Counter(), Counter(), Counter()
    labels, raw_atoms = {}, {}
    for _, row in speech.iterrows():
        votes, atoms = Counter(), []
        for label in row[["E1", "E2", "E3", "E4", "E5", "E6"]]:
            if pd.isna(label):
                continue
            for atom in str(label).split(":"):
                atom = atom.strip()
                if not atom:
                    continue
                atoms.append(atom)
                key = next((k for k, s in zip(["E","U","N","DROP"], sets.values()) if atom in s), None)
                if key:
                    votes[key] += 1
                else:
                    unknown[atom] += 1
        total = sum(votes[k] for k in "EUN")
        if total == 0:
            drops["no_EUN"] += 1
            continue
        if votes["DROP"] > total:
            drops["DROP_dominates"] += 1
            continue
        ordered = {k: v for k, v in votes.items() if k in "EUN"}
        label = max(ordered, key=ordered.get)
        tied = [k for k in "EUN" if votes[k] == votes[label]]
        if len(tied) > 1:
            ties["/".join(tied)] += 1
        labels[row["Name"]] = label
        raw_atoms[row["Name"]] = set(atoms)
    assert not unknown
    path = record(ROOT / "data/nnime_crab_format.csv")
    rows = list(csv.DictReader(path.open()))
    counts, seconds = defaultdict(Counter), defaultdict(list)
    groups = defaultdict(set)
    families = {
        "悲傷／低落": sets["UNCONFIDENT"] & {"傷心","低落","失望","愧疚","無奈","痛","痛苦"},
        "挫折／疑惑": {"挫折","傻眼","懷疑","疑惑","莫名奇妙"},
        "恐懼／焦慮": {"恐懼","緊張","緊張，惶恐","焦慮","焦急","著急","擔心","擔心，下決心","憂心","心虛","不甘心","心不甘","急","急澄清","壓力"},
        "尷尬／低聲下氣": {"尷尬","囧","害躁","低聲下氣"},
    }
    family_counts = Counter()
    for row in rows:
        name = Path(row["FileName"]).stem
        cl = next(k for k, col in zip("EUN", CLASSES) if float(row[col]) == 1)
        assert labels[name] == cl
        team = int(name.split("_")[1])
        expected = "Train" if team in {1,2,3,5,6,7,8,9,10,11,12,13,14,15,16} else "Development" if team in {17,18,19} else "Test" if team in {4,20,21,22} else "UNK"
        assert row["Split_Set"] == expected
        counts[expected][cl] += 1
        groups[expected].add(team)
        with wave.open(row["FileName"]) as wav:
            duration = wav.getnframes() / wav.getframerate()
        seconds[(expected, cl)].append(duration)
        if expected == "Train" and cl == "U":
            for family, atoms in families.items():
                family_counts[family] += bool(raw_atoms[name] & atoms)
    assert len(rows) == len(labels) == 4300
    assert [counts["Train"][k] for k in "EUN"] == [808,628,1618]
    assert sum(ties.values()) == 275
    durations = {}
    for (split, cl), vals in seconds.items():
        durations[f"{split}/{cl}"] = dict(n=len(vals), minutes=sum(vals)/60,
            mean=statistics.mean(vals), median=statistics.median(vals),
            min=min(vals), max=max(vals), under1=sum(v<1 for v in vals),
            over12=sum(v>12 for v in vals))
    v1 = pd.read_csv(record(ROOT / "data/bilingual_strategyA.csv"))
    v1train = v1[v1.Split_Set == "Train"]
    v1counts = v1train.groupby("Language")[CLASSES].sum().astype(int).to_dict("index")
    assert sum(sum(x.values()) for x in v1counts.values()) == 41744
    return dict(input_rows=len(raw), speech_rows=len(speech),
        non_speech=len(raw)-len(speech), drop=dict(drops), ties=dict(ties),
        mapping_labels={k:sorted(v) for k,v in sets.items()},
        counts={k:dict(v) for k,v in counts.items()}, duration=durations,
        families=dict(family_counts), teams={k:sorted(v) for k,v in groups.items()},
        v1=v1counts, real_effective_seconds=sum(min(t,12) for (s,c),vs in seconds.items() if s=="Train" for t in vs))


def audit_results():
    runs, tests = {}, {}
    campaigns = ["n3_phase_a_long12_20260909", "n3_a25_cl_ablation_20260911",
                 "n3_phase_a_multiseed_20260911", "n3_phase_b_matchmin1_s42_20260912"]
    for campaign in campaigns:
        for path in sorted((ROOT / "experiments" / campaign).glob("*/run_result.json")):
            item = read_json(path)
            logs = list(path.parent.glob("logging_*.txt"))
            assert len(logs) == 1
            text = record(logs[0]).read_text()
            chunks = re.split(r"\nEpoch (\d+)\n", text)
            epochs = {int(chunks[i]):chunks[i+1] for i in range(1,len(chunks),2)}
            best = epochs[item["best_epoch"]]
            per = {}
            for cl in CLASSES:
                m = re.search(r"^\s*" + cl + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", best, re.M)
                assert m, (path, cl)
                per[cl] = dict(zip(["precision","recall","f1","support"], map(float,m.groups())))
            item.update(campaign=campaign, actual_epochs=len(epochs), per_class_rounded=per)
            runs[path.parent.name] = item
    for campaign in ["n3_phase_a_test_20260910", "n3_phase_b_test_s42_20260914"]:
        for path in sorted((ROOT/"experiments"/campaign).glob("*/test_result.json")):
            item = read_json(path)
            cm = np.asarray(item["confusion_matrix"])
            assert cm.sum() == 638 and cm.sum(1).tolist() == [181,165,292]
            f1 = 2*np.diag(cm)/(cm.sum(0)+cm.sum(1))
            assert abs(f1.mean()-item["test_macro_f1"]) < 1e-12
            for i,cl in enumerate(CLASSES):
                assert abs(f1[i]-item["per_class"][cl]["f1"]) < 1e-12
            tests[path.parent.name] = item
    assert len(runs)==19 and len(tests)==7
    agg = read_json(ROOT/"experiments/n3_phase_a_multiseed_20260911/aggregate_dev.json")
    for dose in [25,50,75]:
        vals = [runs[f"n3a{dose}s{s}"]["best_dev_macro_f1"] for s in [42,43,44]]
        assert abs(statistics.mean(vals)-agg["by_dose"][str(dose)]["mean"])<1e-12
    return runs,tests,agg


def make_tables(d, runs, tests, agg, b, am):
    pct = lambda x:f"{100*x:.2f}%"
    table("splits",["資料","E 興奮","U 不自信","N 中性","總數"],
        [[s]+[f'{d["counts"][k][c]:,} / {100*d["counts"][k][c]/sum(d["counts"][k].values()):.2f}%' for c in "EUN"]+[f'{sum(d["counts"][k].values()):,}']
         for s,k in [("Train","Train"),("Dev","Development"),("Test","Test")]])
    table("dose",["組別","新增 U","E","U 合計","N","Train 合計","U 占比","TTS 占比"],
        [[name,n,808,628+n,1618,3054+n,f"{100*(628+n)/(3054+n):.2f}%",f"{100*n/(3054+n):.2f}%"]
         for name,n in [("原 N3",0),("A/B25",157),("A/B50",314),("A/B75",471)]])
    table("duration",["劑量","A 平均秒","B 平均秒","A TTS 分鐘","B TTS 分鐘"],
        [[f"{q}%",f'{am[q]["tts_seconds"]/am[q]["n_tts_train"]:.3f}',f'{b["arms"][f"B{q}-min1"]["mean_seconds"]:.3f}',
          f'{am[q]["tts_seconds"]/60:.2f}',f'{b["arms"][f"B{q}-min1"]["total_seconds"]/60:.2f}'] for q in [25,50,75]])
    table("historical",["模型","Dev Macro F1","Test Macro F1","Test E F1","Test U F1","Test N F1"],
        [[label,"52.83%" if key=="n3_reference" else pct(runs[key]["best_dev_macro_f1"]),
          pct(tests[key]["test_macro_f1"])]+[pct(tests[key]["per_class"][cl]["f1"]) for cl in CLASSES]
         for label,key in [("原 N3","n3_reference"),("舊 A25","n3a25"),("舊 A50","n3a50"),("舊 A75","n3a75")]])
    table("cl",["CL 權重","Dev Macro F1","Dev U F1","最佳輪","實跑輪","Test"],
        [[i,pct(runs[f"n3a25cl{i}"]["best_dev_macro_f1"]),
          pct(runs[f"n3a25cl{i}"]["per_class_rounded"]["Unconfident"]["f1"]),
          runs[f"n3a25cl{i}"]["best_epoch"]+1,runs[f"n3a25cl{i}"]["actual_epochs"],"未做"] for i in range(4)])
    table("seeds",["劑量","seed42","seed43","seed44","平均","樣本標準差"],
        [[f"A{q}"]+[pct(v) for v in agg["by_dose"][str(q)]["values"]]
          +[pct(agg["by_dose"][str(q)]["mean"]),f'{100*agg["by_dose"][str(q)]["sample_std"]:.2f} pp'] for q in [25,50,75]])
    table("paired",["劑量","A Dev F1","B Dev F1","B−A","U recall：A → B","抓到 U：A → B"],
        [[f"{q}%",pct(runs[f"n3a{q}s42"]["best_dev_macro_f1"]),pct(runs[f"n3b{q}s42"]["best_dev_macro_f1"]),
          f'{100*(runs[f"n3b{q}s42"]["best_dev_macro_f1"]-runs[f"n3a{q}s42"]["best_dev_macro_f1"]):+.2f} pp',
          " → ".join(pct(runs[f"n3{x}{q}s42"]["per_class_rounded"]["Unconfident"]["recall"]) for x in "ab"),
          " → ".join(str(round(190*runs[f"n3{x}{q}s42"]["per_class_rounded"]["Unconfident"]["recall"])) for x in "ab")] for q in [25,50,75]])
    table("btest",["模型","Dev F1","Test F1","Test E F1","Test U F1","Test N F1","U precision","U recall"],
        [[f"B{q}",pct(runs[f"n3b{q}s42"]["best_dev_macro_f1"]),pct(tests[f"n3b{q}s42"]["test_macro_f1"])]
         +[pct(tests[f"n3b{q}s42"]["per_class"][cl]["f1"]) for cl in CLASSES]
         +[pct(tests[f"n3b{q}s42"]["per_class"]["Unconfident"][m]) for m in ["precision","recall"]] for q in [25,50,75]])
    table("inventory",["Run","Dev Macro F1","最佳／實跑輪","Test Macro F1","W&B ID"],
        [[key,pct(v["best_dev_macro_f1"]),f'{v["best_epoch"]+1} / {v["actual_epochs"]}',
          pct(tests[key]["test_macro_f1"]) if key in tests else "未做",
          v["wandb_url"].rsplit("/",1)[-1]] for key,v in runs.items()])
    table("train_duration",["Train 類別","筆數","原音訊分鐘","平均秒","中位秒","少於 1 秒"],
        [[c, d["duration"][f"Train/{c}"]["n"],f'{d["duration"][f"Train/{c}"]["minutes"]:.2f}',
          f'{d["duration"][f"Train/{c}"]["mean"]:.3f}',f'{d["duration"][f"Train/{c}"]["median"]:.3f}',d["duration"][f"Train/{c}"]["under1"]] for c in "EUN"])


def make_figures(d, runs, tests, agg, b):
    # A complete, code-grounded mapping rather than an "8 classes -> 3" shortcut.
    fig,ax=graph_axes(15,6.5)
    panel(ax,.1,2.35,2.6,1.8,"NNIME 細情緒標註",
          "六位評分者 E1–E6\n每人可填多個標籤\n以「:」拆成原子票",fs=12,title_fs=14)
    items = [
        (5.2,1.1,"E 興奮｜9 個標籤","快樂、喜樂、興奮、驚訝、期待、振作、鬥志、激動、自信",E),
        (2.7,2.25,"U 不自信｜31 個標籤",
         "恐懼／焦慮：恐懼、緊張、緊張，惶恐、焦慮、焦急、著急、擔心、擔心，下決心、\n"
         "　　　　　　憂心、心虛、不甘心、心不甘、急、急澄清、壓力\n"
         "悲傷／低落：傷心、低落、失望、愧疚、無奈、痛、痛苦\n"
         "社交低自信：尷尬、囧、害躁、低聲下氣　　挫折／疑惑：挫折、傻眼、懷疑、疑惑、莫名奇妙",U),
        (1.5,1.0,"N 中性｜6 個標籤","中性、放鬆、祥和、認真、嚴肅、其他",N),
        (.05,1.25,"DROP｜16 個標籤（排除票，不是模型的第 4 類）",
         "不以為然、不屑、不悅、不耐煩、埋怨、煩、煩躁、生氣、警告、無法標記(雜訊，觀眾笑聲)、\n"
         "下定決心、感性、語重心長、關懷、想睡、無聊",GREY)]
    for y,h,title,body,c in items:
        panel(ax,3.15,y,11.6,h,title,body,c,fs=11.3,title_fs=14)
        arrow(ax,(2.72,3.25),(3.12,y+h/2),c)
    save(fig,"nnime_mapping")
    fig,ax=graph_axes(15,4.7)
    panel(ax,.1,2.7,3.05,1.5,"① 原始標註 5,596","先保留 Type = Speech\n非語音事件排除 858",fs=12)
    panel(ax,3.8,2.7,3.4,1.5,"② Speech 4,738","六位標註者拆票\nE／U／N／DROP 分別累計",fs=12)
    panel(ax,7.9,2.7,3.0,1.5,"③ 排除 438","無 E/U/N 票：201\nDROP > E+U+N：237",fs=12)
    panel(ax,11.55,2.7,3.25,1.5,"④ 保留 4,300","E/U/N 中票數最大者\n音訊與既有逐字稿對應",fs=12)
    for x1,x2 in [(3.15,3.8),(7.2,7.9),(10.9,11.55)]:arrow(ax,(x1,3.45),(x2,3.45))
    arrow(ax,(13.2,2.68),(13.2,2.05))
    panel(ax,.1,.2,6.7,1.8,"不是「六人至少四人同意」",
          "例（示意）：緊張:傷心／中性／中性／中性／恐懼／放鬆\n原子票 U=3、N=4 → N；不是先替每人選一個三類標籤。\n平票按現行程式首次遇到的類別決定，未另設仲裁。",fs=11.7)
    panel(ax,7.4,.2,7.4,1.8,"⑤ 依 team 分組，不隨機打散片段",
          "Train：15 組 → 3,054　｜　Dev：3 組 → 608\nTest：4 組 → 638（第 4、22 組放同一 split）\n三類輸出為 one-hot；實際 CSV 4,300 列全部核對一致。",fs=12)
    save(fig,"nnime_pipeline")
    fig,axes=plt.subplots(1,2,figsize=(13,3.3),layout="constrained")
    ax=axes[0]
    splitkeys=["Train","Development","Test"]; left=np.zeros(3)
    for c,col in zip("EUN",COLORS):
        vals=np.array([d["counts"][s][c]/sum(d["counts"][s].values())*100 for s in splitkeys])
        ax.barh(range(3),vals,left=left,color=col,label=c)
        for i,(v,l) in enumerate(zip(vals,left)):ax.text(l+v/2,i,f"{v:.2f}%",ha="center",va="center",color="white",fontsize=12,weight="bold")
        left+=vals
    ax.set_yticks(range(3),["Train · 3,054","Dev · 608","Test · 638"])
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel("各 split 內的比例（%）")
    ax.legend(ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.23))
    ax=axes[1]; vals=[808,628,1618]
    bars=ax.bar(["E 興奮","U 不自信","N 中性"],vals,color=COLORS)
    ax.bar_label(bars,labels=[f"{v:,}" for v in vals],padding=4,fontsize=13)
    ax.set_ylim(0,1900);ax.set_ylabel("NNIME Train 筆數");ax.grid(axis="y",alpha=.15)
    save(fig,"nnime_distribution")
    fig,ax=plt.subplots(figsize=(12.5,2.6),layout="constrained")
    labels=list(d["families"]); vals=list(d["families"].values())
    bars=ax.barh(labels,vals,color=U);ax.invert_yaxis()
    ax.bar_label(bars,labels=[f"{v}/628（{v/628*100:.1f}%）" for v in vals],padding=6)
    ax.set_xlim(0,675);ax.set_xlabel("Train U 中，至少一位評分者提及該標籤家族的片段數（家族可重疊）")
    save(fig,"u_family")
    fig,ax=plt.subplots(figsize=(12,3.4),layout="constrained");x=np.arange(6)
    for i,(label,counts,col) in enumerate([
        ("真人 Train U ≥1 秒",b["reference_counts"],DARK),
        ("B25",b["arms"]["B25-min1"]["actual_bin_counts"],"#B49ACB"),
        ("B50",b["arms"]["B50-min1"]["actual_bin_counts"],U),
        ("B75",b["arms"]["B75-min1"]["actual_bin_counts"],N)]):
        ax.bar(x+(i-1.5)*.2,np.array(counts)/sum(counts)*100,.19,label=label,color=col)
    ax.set_xticks(x,["1–<2","2–<4","4–<6","6–<8","8–<10","10–12"])
    ax.set(xlabel="有效音訊秒數（真人先套 min(原長度, 12)）",ylabel="組內比例（%）",ylim=(0,49))
    ax.legend(ncol=4,frameon=False,loc="upper center");ax.grid(axis="y",alpha=.15)
    save(fig,"duration_match")
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.0),layout="constrained")
    ids=["n3_reference","n3a25","n3a50","n3a75"];names=["原 N3","舊 A25","舊 A50","舊 A75"]
    for ax,key,label in [(axes[0],"dev","Dev Macro F1（%）"),(axes[1],"test","Test Macro F1（%）")]:
        vals=[52.83 if id=="n3_reference" else runs[id]["best_dev_macro_f1"]*100 for id in ids] if key=="dev" else [tests[id]["test_macro_f1"]*100 for id in ids]
        bars=ax.bar(names,vals,color=[GREY,N,N,N]);ax.bar_label(bars,fmt="%.2f",padding=3)
        ax.set(ylim=(0,70),ylabel=label);ax.grid(axis="y",alpha=.15)
    save(fig,"historical_results")
    fig,ax=plt.subplots(figsize=(10,2.6),layout="constrained")
    vals=[runs[f"n3a25cl{i}"]["best_dev_macro_f1"]*100 for i in range(4)]
    bars=ax.bar(["CL0","CL1","CL2","CL3"],vals,color=[GREY,"#8CBEC4",N,"#6295AB"])
    ax.bar_label(bars,fmt="%.2f",padding=4);ax.set(ylim=(0,65),ylabel="Dev Macro F1（%）")
    ax.grid(axis="y",alpha=.15);save(fig,"cl_results")
    fig,ax=plt.subplots(figsize=(11,3.2),layout="constrained")
    for i,q in enumerate([25,50,75]):
        vals=np.array(agg["by_dose"][str(q)]["values"])*100
        for j,s in enumerate([42,43,44]):
            ax.scatter(i+(j-1)*.08,vals[j],s=60,marker=["o","s","^"][j],color=[N,E,U][j],label=f"seed{s}" if i==0 else None,zorder=4)
        ax.errorbar(i+.25,vals.mean(),yerr=vals.std(ddof=1),fmt="D",capsize=6,color=DARK,label="平均 ± 樣本標準差" if i==0 else None)
    ax.set_xticks(range(3),["A25","A50","A75"]);ax.set(ylabel="Dev Macro F1（%）",ylim=(51,59))
    ax.grid(axis="y",alpha=.2);ax.legend(frameon=False,ncol=4,loc="upper center",fontsize=10)
    save(fig,"multiseed_results")
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.1),layout="constrained")
    for ax,metric,label,ylim in [(axes[0],"macro","Dev Macro F1（%）",70),(axes[1],"recall","Dev U recall（%）",70)]:
        x=np.arange(3)
        for off,group,color in [(-.18,"a",N),(.18,"b",U)]:
            vals=[(runs[f"n3{group}{q}s42"]["best_dev_macro_f1"] if metric=="macro" else runs[f"n3{group}{q}s42"]["per_class_rounded"]["Unconfident"]["recall"])*100 for q in [25,50,75]]
            bars=ax.bar(x+off,vals,.34,color=color,label="A Long12" if group=="a" else "B Match-min1")
            ax.bar_label(bars,fmt="%.2f",fontsize=10,padding=3)
        ax.set_xticks(x,["25%","50%","75%"]);ax.set(ylim=(0,ylim),ylabel=label)
        ax.legend(frameon=False,ncol=2,loc="upper center",fontsize=10);ax.grid(axis="y",alpha=.15)
    save(fig,"paired_ab")
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.0),layout="constrained")
    for ax,q in zip(axes,[25,50,75]):
        cm=np.array(tests[f"n3b{q}s42"]["confusion_matrix"])
        ax.imshow(cm/cm.sum(1,keepdims=True)*100,cmap="Blues",vmin=0,vmax=100)
        for i in range(3):
            for j in range(3):
                pc=cm[i,j]/cm[i].sum()*100
                ax.text(j,i,f"{cm[i,j]}\n{pc:.1f}%",ha="center",va="center",color="white" if pc>55 else DARK,fontsize=11)
        ax.set_xticks(range(3),["E","U","N"]);ax.set_yticks(range(3),["E","U","N"])
        ax.set(title=f"B{q} · Test",xlabel="預測類別",ylabel="真實類別")
    save(fig,"b_confusion")


def main():
    ASSETS.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    font=fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"font.size":12,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.titleweight":"bold","text.color":DARK,"axes.labelcolor":DARK,
        "pdf.fonttype":3,"svg.fonttype":"path","axes.unicode_minus":False})
    d=audit_data(); runs,tests,agg=audit_results()
    b=read_json(ROOT/"experiments/datasets/nnime_u_match_min1_20260911/build_report.json")
    am={q:read_json(ROOT/f"experiments/n3_phase_a_long12_20260909/n3a{q}_metadata.json") for q in [25,50,75]}
    assert abs(d["real_effective_seconds"]-6765.452)<1e-6
    make_tables(d,runs,tests,agg,b,am)
    make_figures(d,runs,tests,agg,b)
    evidence=dict(generated_at=datetime.now(timezone.utc).isoformat(),
        latest_evaluated_at=max(x["evaluated_at"] for x in tests.values()),
        dataset=d,runs=runs,tests=tests,multiseed=agg,
        sources_sha256=SOURCES,
        scope="aggregate local evidence; no new training/inference/human QC")
    (OUT/"evidence_summary.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"runs":len(runs),"test_checkpoints":len(tests),"nnime_rows":4300,
        "mapping_ties":sum(d["ties"].values()),"u_families":d["families"],
        "sources_hashed":len(SOURCES),"figures":len(list(ASSETS.glob("*.pdf")))},ensure_ascii=False))


if __name__=="__main__":
    main()
