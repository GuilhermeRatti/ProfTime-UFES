#!/usr/bin/env python3
"""
Gera uma figura estática (PNG) da timeline do ProfTime, no estilo das
figuras do artigo ProfInfer: uma linha por thread com os OPERADORES que ela
executa (MUL_MAT, SOFT_MAX, ...) coloridos sobre os estados de escalonamento
(Running / Runnable / Idle), e a trilha de chamadas a llama_decode no topo.

Complementa o Perfetto (interativo): a figura estática serve para relatórios.

Uso:
    python3 proftime_plot.py trace.json                     # trace inteiro
    python3 proftime_plot.py trace.json --start 2 --end 5   # zoom (segundos)
    python3 proftime_plot.py trace.json --op-chart ops.png  # gargalo por operador

Não precisa de root: só lê o JSON gerado pelo proftime.py.
"""

import argparse
import json
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
STYLE = {  # estado -> (cor, altura da barra)
    "Running":  ("#008300", 0.72),
    "Runnable": ("#c98500", 0.44),
    "Idle":     ("#dddbd3", 0.10),  # neutro de propósito: "nada acontecendo"
}
DECODE_COLOR = "#52514e"
# Cores dos operadores: os 6 com mais tempo de CPU ganham um slot, na ordem;
# o resto vira "outros". Verde fica reservado para "Running sem operador".
OP_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#e87ba4", "#eb6834"]
OP_OTHER = "#898781"
OP_TRACK_OFFSET = 100_000_000  # mesmo valor do proftime.py


def load_trace(path):
    """Lê o Chrome Trace JSON do proftime.py e devolve fatias por thread."""
    with open(path) as f:
        events = json.load(f)["traceEvents"]

    names = {}  # tid -> nome da thread (metadados "M")
    threads = {}  # tid -> [(t0_s, dur_s, estado)]
    ops = {}  # tid -> [(t0_s, dur_s, operador)]
    decodes = []  # [(t0_s, dur_s, rótulo)]
    cpus = {}  # cpu -> [(t0_s, dur_s, (tid, comm, é_nossa?))]
    pid = None

    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            names[e["tid"]] = e["args"]["name"]
        elif e.get("ph") == "X":
            t0, dur = e["ts"] / 1e6, e["dur"] / 1e6  # µs -> s
            if e.get("cat") == "token":
                pid = e["pid"]
                decodes.append((t0, dur, e["name"]))
            elif e.get("cat") == "sched":
                pid = e["pid"]
                state = "Running" if e["name"].startswith("Running") else e["name"]
                threads.setdefault(e["tid"], []).append((t0, dur, state))
            elif e.get("cat") == "op":
                tid = e["tid"] - OP_TRACK_OFFSET
                ops.setdefault(tid, []).append((t0, dur, e["name"]))
            elif e.get("cat") == "cpu":
                m = re.match(r"(.*) \[(\d+)\]$", e["name"])
                comm, tid = (m.group(1), int(m.group(2))) if m else (e["name"], 0)
                cpus.setdefault(e["tid"], []).append(
                    (t0, dur, (tid, comm, e["args"].get("ours", False))))
    return pid, names, threads, ops, decodes, cpus


def op_colors(ops):
    """Os 6 operadores com mais tempo total ganham cor fixa, o resto é 'outros'."""
    totals = defaultdict(float)
    for spans in ops.values():
        for _t0, dur, op in spans:
            totals[op] += dur
    top = [op for op, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:len(OP_PALETTE)]]
    return {op: OP_PALETTE[i] for i, op in enumerate(top)}, totals


def plot_op_chart(totals, path):
    """Gráfico de barras: onde a CPU gastou o tempo, por operador."""
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    top, rest = ranked[:10], ranked[10:]
    if rest:
        top.append(("(outros)", sum(v for _, v in rest)))
    total = sum(totals.values())
    names = [op for op, _ in top][::-1]
    vals = [v for _, v in top][::-1]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(top) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(names, vals, height=0.62, color=OP_PALETTE[0], zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.01 * max(vals), bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} s ({100 * v / total:.1f}%)", va="center",
                fontsize=8, color=INK_2)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_xlabel("tempo de CPU somado entre threads (s)", fontsize=9, color=INK_MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=INK_MUTED)
    ax.tick_params(axis="y", labelsize=8, colors=INK_2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_title("Onde está o gargalo? — tempo de CPU por operador",
                 loc="left", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    print(f"figura salva em {path}")


def plot_cpu_view(cpus, pid, start, end, path):
    """
    A visão da Figura 8 do artigo: uma linha por núcleo de CPU, mostrando qual
    thread ocupa cada núcleo ao longo do tempo. Threads da inferência ganham
    cores; o resto do sistema é cinza; vazio = núcleo ocioso.
    """
    cpus = {c: clip(sl, start, end) for c, sl in cpus.items()}
    cpus = {c: sl for c, sl in cpus.items() if sl}
    if not cpus:
        sys.exit("o trace não tem eventos de ocupação de CPU (rode sem --no-cpu-view)")

    # Threads mais ativas ganham as primeiras cores (main sempre primeiro);
    # além dos 6 slots, as restantes compartilham o violeta.
    busy = defaultdict(float)
    for sl in cpus.values():
        for _t, dur, (tid, _c, mine) in sl:
            if mine:
                busy[tid] += dur
    ours = sorted(busy, key=lambda t: (t != pid, -busy[t]))
    color_of = {tid: OP_PALETTE[i] if i < len(OP_PALETTE) else "#4a3aa7"
                for i, tid in enumerate(ours)}

    rows = sorted(cpus)
    fig_h = 0.6 + 0.34 * len(rows)
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for row, cpu in enumerate(rows):
        y = len(rows) - 1 - row
        by_color = defaultdict(list)
        for t0, dur, (tid, _comm, mine) in cpus[cpu]:
            by_color[color_of[tid] if mine else "#c3c2b7"].append((t0, dur))
        for color, spans in by_color.items():
            ax.broken_barh(spans, (y - 0.36, 0.72), facecolors=color,
                           linewidth=0, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"CPU {c}" for c in reversed(rows)], fontsize=8, color=INK_2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(start, end)
    ax.set_xlabel("tempo (s)", fontsize=9, color=INK_MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=INK_MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_title("ProfTime — ocupação dos núcleos de CPU",
                 loc="left", fontsize=11, color=INK, pad=28)

    handles = [Patch(facecolor=color_of[tid],
                     label=("main" if tid == pid else "worker") + f" [{tid}]")
               for tid in ours]
    handles.append(Patch(facecolor="#c3c2b7", label="outros processos"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.005),
              ncol=min(5, len(handles)), frameon=False, fontsize=8,
              handlelength=1.4, handleheight=0.9, labelcolor=INK_2, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    print(f"figura salva em {path} ({len(rows)} CPUs)")


def clip(slices, start, end):
    """Recorta as fatias para a janela [start, end]."""
    out = []
    for t0, dur, label in slices:
        a, b = max(t0, start), min(t0 + dur, end)
        if b > a:
            out.append((a, b - a, label))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", help="trace.json gerado pelo proftime.py")
    p.add_argument("-o", "--output", default="proftime.png")
    p.add_argument("--start", type=float, default=None, help="início da janela (s)")
    p.add_argument("--end", type=float, default=None, help="fim da janela (s)")
    p.add_argument("--op-chart", default=None, metavar="PNG",
                   help="também gera o gráfico de tempo de CPU por operador")
    p.add_argument("--cpus", default=None, metavar="PNG",
                   help="também gera a visão de ocupação por núcleo de CPU")
    args = p.parse_args()

    pid, names, threads, ops, decodes, cpus = load_trace(args.trace)
    if not threads:
        sys.exit("nenhuma fatia de escalonamento no trace")

    colors, totals = op_colors(ops)
    if args.op_chart:
        if not totals:
            sys.exit("o trace não tem eventos de operador (compile o llama.cpp da fonte)")
        plot_op_chart(totals, args.op_chart)

    t_min = min(s[0] for sl in threads.values() for s in sl)
    t_max = max(s[0] + s[1] for sl in threads.values() for s in sl)
    start = args.start if args.start is not None else t_min
    end = args.end if args.end is not None else t_max

    if args.cpus:
        plot_cpu_view(cpus, pid, start, end, args.cpus)

    threads = {tid: clip(sl, start, end) for tid, sl in threads.items()}
    threads = {tid: sl for tid, sl in threads.items() if sl}
    ops = {tid: clip(sl, start, end) for tid, sl in ops.items()}
    decodes = clip(decodes, start, end)

    # Linhas: tokens no topo, depois a thread principal, depois as demais.
    order = sorted(threads, key=lambda tid: (tid != pid, tid))
    rows = ([("tokens", None)] if decodes else []) + [(tid, threads[tid]) for tid in order]

    fig_h = 0.6 + 0.34 * len(rows)
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    labels = []
    for row, (tid, slices) in enumerate(rows):
        y = len(rows) - 1 - row  # topo primeiro
        if tid == "tokens":
            labels.append((y, "llama_decode"))
            ax.broken_barh([(t0, dur) for t0, dur, _ in decodes], (y - 0.36, 0.72),
                           facecolors=DECODE_COLOR, linewidth=0, zorder=3)
            # rótulo direto no prefill, se houver espaço
            t0, dur, label = decodes[0]
            if "prefill" in label and dur > 0.04 * (end - start):
                ax.text(t0 + dur / 2, y, "prefill", ha="center", va="center",
                        fontsize=7, color="#ffffff", zorder=4)
        else:
            who = "main" if tid == pid else "worker"
            labels.append((y, f"{who} [{tid}]"))
            for state, (color, height) in STYLE.items():
                spans = [(t0, dur) for t0, dur, s in slices if s == state]
                if spans:
                    ax.broken_barh(spans, (y - height / 2, height),
                                   facecolors=color, linewidth=0, zorder=3)
            # operadores por cima do estado Running: o verde que sobrar
            # aparecendo é CPU ocupada SEM operador (spin/sincronização).
            # Agrupados por cor: UMA chamada de broken_barh desenha milhares
            # de fatias (uma por fatia travaria o matplotlib).
            by_color = defaultdict(list)
            for t0, dur, op in ops.get(tid, []):
                by_color[colors.get(op, OP_OTHER)].append((t0, dur))
            for color, spans in by_color.items():
                ax.broken_barh(spans, (y - 0.36, 0.72), linewidth=0,
                               facecolors=color, zorder=4)

    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([name for _, name in labels], fontsize=8, color=INK_2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(start, end)
    ax.set_xlabel("tempo (s)", fontsize=9, color=INK_MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=INK_MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    title = ("ProfTime — operadores e estados das threads da inferência"
             if colors else "ProfTime — estados das threads da inferência")
    handles = [Patch(facecolor=c, label=op) for op, c in colors.items()]
    if any(op not in colors for spans in ops.values() for _t0, _d, op in spans):
        handles.append(Patch(facecolor=OP_OTHER, label="outros ops"))
    handles += [Patch(facecolor=STYLE["Running"][0],
                      label="Running sem operador" if colors else "Running"),
                Patch(facecolor=STYLE["Runnable"][0], label="Runnable (esperando CPU)"),
                Patch(facecolor=STYLE["Idle"][0], label="Idle (dormindo)"),
                Patch(facecolor=DECODE_COLOR, label="llama_decode")]
    legend_rows = -(-len(handles) // 5)  # ceil: o pad do título depende disso
    ax.set_title(title, loc="left", fontsize=11, color=INK,
                 pad=14 * legend_rows + 14)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.005),
              ncol=5, frameon=False, fontsize=8, handlelength=1.4,
              handleheight=0.9, labelcolor=INK_2, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(args.output, facecolor=SURFACE, bbox_inches="tight")
    janela = f"{start:.2f}s–{end:.2f}s" if (args.start or args.end) else "trace completo"
    print(f"figura salva em {args.output} ({janela}, {len(rows)} linhas)")


if __name__ == "__main__":
    main()
