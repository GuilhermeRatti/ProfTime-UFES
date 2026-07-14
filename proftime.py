#!/usr/bin/env python3
"""
ProfTime — timeline de threads de uma inferência LLM, via eBPF.

Reimplementação didática da visão "ProfTime" do artigo ProfInfer
(arXiv:2601.20755): rastreamos as threads do llama.cpp com tracepoints do
escalonador do Linux (sched_switch / sched_wakeup) e marcamos cada passo de
inferência com um uprobe na função llama_decode — tudo sem modificar ou
recompilar o llama.cpp.

O resultado é um arquivo JSON no Chrome Trace Event Format, que pode ser
aberto em https://ui.perfetto.dev para visualizar a linha do tempo.

Uso (precisa de root, como todo programa eBPF):

    sudo python3 proftime.py -o trace.json -- \
        ./llama.cpp/llama-cli -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
        -p "Explique o que é eBPF." -n 64 -t 4 --no-warmup -no-cnv

Fluxo geral:

    1. Inicia o comando de inferência PAUSADO (fork + SIGSTOP antes do exec);
    2. Compila e carrega o programa eBPF (proftime_bpf.c) com o PID alvo;
    3. Anexa os probes e registra a thread principal no mapa `interest`;
    4. Libera o processo (SIGCONT) e coleta eventos do ring buffer;
    5. Ao final, reconstrói os estados de cada thread (Running / Runnable /
       Idle) e grava o JSON da timeline.

Pausar o processo antes do exec garante que nenhum evento é perdido: quando
o llama.cpp criar suas threads de trabalho, os probes já estarão ativos.
(Usamos fork/exec diretamente, e não subprocess.Popen, porque o Popen só
retorna depois do exec do filho — com o filho parado antes do exec, seria
um deadlock.)
"""

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from ggml_ops import GGML_OPS  # índice = valor do enum ggml_op (gerado do ggml.h)

# Mesmos valores do enum ev_kind em proftime_bpf.c
EV_SWITCH_IN = 0
EV_SWITCH_OUT = 1
EV_WAKEUP = 2
EV_DECODE_BEGIN = 3
EV_DECODE_END = 4
EV_CPU = 5

# Os três estados de thread do ProfTime (artigo, seção 4.2)
RUNNING = "Running"    # executando em alguma CPU
RUNNABLE = "Runnable"  # pronta para executar, mas esperando por uma CPU
IDLE = "Idle"          # bloqueada/dormindo (ex.: esperando trabalho num futex)

# tid sintético usado na timeline para a trilha de tokens (llama_decode)
TOKENS_TRACK_TID = 0


def parse_args():
    if "--" not in sys.argv:
        sys.exit("uso: sudo python3 proftime.py [-o saida.json] -- <comando de inferência...>")
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="proftime_trace.json",
                        help="arquivo de saída (Chrome Trace Event Format)")
    parser.add_argument("--llama-lib", default=None,
                        help="caminho para libllama.so (padrão: procura ao lado do executável)")
    parser.add_argument("--ggml-lib", default=None,
                        help="caminho para libggml-cpu.so (padrão: procura ao lado do executável)")
    parser.add_argument("--no-cpu-view", action="store_true",
                        help="não rastrear a ocupação de cada CPU por outros processos")
    args = parser.parse_args(sys.argv[1:split])
    workload = sys.argv[split + 1:]
    if not workload:
        sys.exit("erro: nenhum comando de inferência após '--'")
    return args, workload


def find_lib(workload_cmd, libname, explicit_path):
    """Localiza uma biblioteca do llama.cpp ao lado do executável do workload."""
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else sys.exit(f"erro: {explicit_path} não existe")
    exe = Path(workload_cmd[0]).resolve()
    for candidate in [exe.parent / libname, Path("llama.cpp") / libname]:
        if candidate.exists():
            return candidate.resolve()
    return None


def attach_op_probes(bpf, libggml):
    """
    Anexa uprobe+uretprobe a cada função de operador do backend de CPU.

    O dispatcher (ggml_compute_forward) é `static` e é inlined pelo compilador,
    então não dá para sondá-lo. Mas cada operador tem sua própria função
    não-estática — ggml_compute_forward_mul_mat, _soft_max, ... — e todas têm
    a mesma assinatura (params, tensor). Listamos os símbolos da biblioteca
    com `nm` e anexamos só os que correspondem a um valor do enum ggml_op
    (isso exclui funções auxiliares internas, como as de cada ativação da
    UNARY, que causariam dupla contagem).
    """
    out = subprocess.run(["nm", "--defined-only", str(libggml)],
                         capture_output=True, text=True).stdout
    in_lib = set(re.findall(r" [TtWw] (ggml_compute_forward_\w+)$", out, re.M))
    wanted = {f"ggml_compute_forward_{op.lower()}" for op in GGML_OPS}
    syms = sorted(in_lib & wanted)
    for sym in syms:
        bpf.attach_uprobe(name=str(libggml), sym=sym, fn_name="on_op_begin")
        bpf.attach_uretprobe(name=str(libggml), sym=sym, fn_name="on_op_end")
    return syms


def spawn_paused(cmd):
    """
    Inicia `cmd` num processo filho que fica PAUSADO antes do exec.

    Truque clássico: o filho manda SIGSTOP para si mesmo logo após o fork;
    o pai espera o filho parar (waitpid + WUNTRACED), prepara os probes com
    calma e só então o libera com SIGCONT.
    """
    pid = os.fork()
    if pid == 0:  # filho
        os.kill(os.getpid(), signal.SIGSTOP)  # pausa até o SIGCONT do pai
        try:
            os.execvp(cmd[0], cmd)  # substitui o filho pelo comando
        except OSError as e:
            print(f"[proftime] erro ao executar {cmd[0]}: {e}", file=sys.stderr)
            os._exit(127)
    os.waitpid(pid, os.WUNTRACED)  # espera o filho parar de fato
    return pid


def collect_events(bpf, workload_pid):
    """Lê eventos dos ring buffers até o processo de inferência terminar."""
    raw = []
    raw_ops = []

    def on_event(ctx, data, size):
        ev = bpf["events"].event(data)
        raw.append((ev.ts, ev.kind, ev.tid, ev.cpu, ev.prev_state,
                    ev.comm.decode("utf-8", errors="replace").strip("\x00")))

    def on_op_event(ctx, data, size):
        ev = bpf["op_events"].event(data)
        op = GGML_OPS[ev.op] if ev.op < len(GGML_OPS) else f"OP_{ev.op}"
        raw_ops.append((ev.ts, ev.dur, ev.tid, op,
                        ev.tname.decode("utf-8", errors="replace").strip("\x00")))

    bpf["events"].open_ring_buffer(on_event)
    bpf["op_events"].open_ring_buffer(on_op_event)

    # Só agora, com tudo pronto, o processo pausado é liberado.
    os.kill(workload_pid, signal.SIGCONT)
    print(f"[proftime] inferência liberada (pid {workload_pid}); coletando eventos...\n")

    try:
        while True:
            bpf.ring_buffer_poll(20)
            wpid, status = os.waitpid(workload_pid, os.WNOHANG)
            if wpid == workload_pid and (os.WIFEXITED(status) or os.WIFSIGNALED(status)):
                break
        # O processo terminou; drena o que restou nos buffers.
        for _ in range(10):
            bpf.ring_buffer_poll(50)
    except KeyboardInterrupt:
        print("\n[proftime] interrompido; encerrando a inferência...")
        try:
            os.kill(workload_pid, signal.SIGTERM)
            os.waitpid(workload_pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass  # o processo já tinha terminado

    return raw, raw_ops


def build_timeline(raw_events):
    """
    Reconstrói, a partir dos eventos brutos, os intervalos de estado de cada
    thread. Máquina de estados por thread (artigo, seção 4.2):

        SWITCH_IN            -> Running (na CPU do evento)
        SWITCH_OUT, state==0 -> Runnable (perdeu a CPU, mas ainda quer rodar)
        SWITCH_OUT, state!=0 -> Idle (bloqueou voluntariamente, ex.: futex)
        WAKEUP               -> Runnable (acordou; falta o escalonador dar CPU)
    """
    threads = {}  # tid -> {"comm", "state", "since", "cpu", "slices"}
    decodes = []  # [(ts_inicio, ts_fim), ...] das chamadas a llama_decode
    open_decode = {}  # tid -> ts_inicio

    def switch_state(tid, ts, new_state, cpu, comm):
        th = threads.setdefault(tid, {"comm": comm, "state": None,
                                      "since": ts, "cpu": None, "slices": []})
        if comm:
            th["comm"] = comm
        if th["state"] is not None:
            th["slices"].append((th["since"], ts, th["state"], th["cpu"]))
        th["state"], th["since"], th["cpu"] = new_state, ts, cpu

    last_ts = 0
    for ts, kind, tid, cpu, prev_state, comm in sorted(raw_events):
        last_ts = max(last_ts, ts)
        if kind == EV_SWITCH_IN:
            switch_state(tid, ts, RUNNING, cpu, comm)
        elif kind == EV_SWITCH_OUT:
            switch_state(tid, ts, RUNNABLE if prev_state == 0 else IDLE, None, comm)
        elif kind == EV_WAKEUP:
            # Só transiciona se estava dormindo; wakeups redundantes são ignorados.
            if tid not in threads or threads[tid]["state"] == IDLE:
                switch_state(tid, ts, RUNNABLE, None, comm)
        elif kind == EV_DECODE_BEGIN:
            open_decode[tid] = ts
        elif kind == EV_DECODE_END and tid in open_decode:
            decodes.append((open_decode.pop(tid), ts))

    # Fecha os intervalos ainda abertos no último timestamp visto.
    for th in threads.values():
        if th["state"] is not None and last_ts > th["since"]:
            th["slices"].append((th["since"], last_ts, th["state"], th["cpu"]))

    return threads, decodes


def build_cpu_lanes(raw_events):
    """
    Ocupação de cada CPU ao longo do tempo, a partir dos eventos EV_CPU:
    cada sched_switch diz quem ENTROU na CPU; o ocupante anterior sai.
    tid 0 é a "swapper" (thread ociosa do kernel) = CPU livre.
    """
    lanes = defaultdict(list)  # cpu -> [(t0, t1, tid, comm)]
    current = {}               # cpu -> (t0, tid, comm)
    last_ts = 0
    for ts, kind, tid, cpu, _prev, comm in sorted(raw_events):
        if kind != EV_CPU:
            continue
        last_ts = max(last_ts, ts)
        if cpu in current:
            t0, ptid, pcomm = current[cpu]
            if ptid != 0 and ts > t0:  # ignora os períodos ociosos
                lanes[cpu].append((t0, ts, ptid, pcomm))
        current[cpu] = (ts, tid, comm)
    for cpu, (t0, tid, comm) in current.items():
        if tid != 0 and last_ts > t0:
            lanes[cpu].append((t0, last_ts, tid, comm))
    return lanes


OP_TRACK_OFFSET = 100_000_000  # tid sintético da trilha de operadores de cada thread
CPU_TRACK_PID = 0              # "processo" sintético que agrupa as trilhas de CPU


def write_chrome_trace(path, pid, proc_name, threads, decodes, op_spans, cpu_lanes):
    """Converte os intervalos para o Chrome Trace Event Format (ts/dur em µs)."""
    all_ts = [s[0] for th in threads.values() for s in th["slices"]]
    all_ts += [d[0] for d in decodes]
    t0 = min(all_ts) if all_ts else 0
    us = lambda ns: (ns - t0) / 1000.0

    events = [
        {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": f"{proc_name} (ProfTime)"}},
        {"ph": "M", "pid": pid, "tid": TOKENS_TRACK_TID, "name": "thread_name",
         "args": {"name": "tokens (llama_decode)"}},
        {"ph": "M", "pid": pid, "tid": TOKENS_TRACK_TID, "name": "thread_sort_index",
         "args": {"sort_index": -1}},
    ]

    for i, tid in enumerate(sorted(threads)):
        th = threads[tid]
        events.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name",
                       "args": {"name": f"{th['comm'] or 'thread'} [{tid}]"}})
        events.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_sort_index",
                       "args": {"sort_index": 2 * i}})
        for start, end, state, cpu in th["slices"]:
            name = f"Running (CPU {cpu})" if state == RUNNING else state
            events.append({"ph": "X", "cat": "sched", "pid": pid, "tid": tid,
                           "ts": us(start), "dur": (end - start) / 1000.0,
                           "name": name})
        # Trilha "ops" logo abaixo da trilha de estados da mesma thread.
        if tid in op_spans:
            events.append({"ph": "M", "pid": pid, "tid": OP_TRACK_OFFSET + tid,
                           "name": "thread_name", "args": {"name": f"ops [{tid}]"}})
            events.append({"ph": "M", "pid": pid, "tid": OP_TRACK_OFFSET + tid,
                           "name": "thread_sort_index", "args": {"sort_index": 2 * i + 1}})
            for start, dur, op, tname in op_spans[tid]:
                events.append({"ph": "X", "cat": "op", "pid": pid,
                               "tid": OP_TRACK_OFFSET + tid,
                               "ts": us(start), "dur": dur / 1000.0,
                               "name": op, "args": {"tensor": tname}})

    for i, (start, end) in enumerate(decodes):
        label = "prefill" if i == 0 else f"token {i}"
        events.append({"ph": "X", "cat": "token", "pid": pid, "tid": TOKENS_TRACK_TID,
                       "ts": us(start), "dur": (end - start) / 1000.0,
                       "name": f"llama_decode ({label})"})

    # Ocupação por núcleo: uma trilha por CPU, com qualquer processo.
    if cpu_lanes:
        events.append({"ph": "M", "pid": CPU_TRACK_PID, "name": "process_name",
                       "args": {"name": "CPUs (ocupação por núcleo)"}})
        for cpu, spans in sorted(cpu_lanes.items()):
            events.append({"ph": "M", "pid": CPU_TRACK_PID, "tid": cpu,
                           "name": "thread_name", "args": {"name": f"CPU {cpu}"}})
            events.append({"ph": "M", "pid": CPU_TRACK_PID, "tid": cpu,
                           "name": "thread_sort_index", "args": {"sort_index": cpu}})
            for start, end, tid, comm in spans:
                events.append({"ph": "X", "cat": "cpu", "pid": CPU_TRACK_PID,
                               "tid": cpu, "ts": us(start),
                               "dur": (end - start) / 1000.0,
                               "name": f"{comm} [{tid}]",
                               "args": {"ours": tid in threads}})

    with open(path, "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)

    # Se rodou com sudo, devolve a posse do arquivo ao usuário original.
    if "SUDO_UID" in os.environ:
        os.chown(path, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))


def print_op_summary(op_spans, threads):
    """A pergunta de ouro: em que a CPU gastou o tempo da inferência?"""
    if not op_spans:
        return
    by_op = defaultdict(lambda: [0, 0])     # op -> [tempo_ns, chamadas]
    by_layer = defaultdict(int)             # camada -> tempo_ns
    for spans in op_spans.values():
        for _ts, dur, op, tname in spans:
            by_op[op][0] += dur
            by_op[op][1] += 1
            m = re.search(r"-(\d+)$", tname)  # ex.: "ffn_out-12" -> camada 12
            if m:
                by_layer[int(m.group(1))] += dur

    total_op = sum(t for t, _ in by_op.values())
    total_running = sum(e - s for th in threads.values()
                        for s, e, state, _ in th["slices"] if state == RUNNING)

    print("\n===== Onde está o gargalo? (tempo de CPU somado entre threads) =====")
    ranked = sorted(by_op.items(), key=lambda kv: -kv[1][0])
    for op, (t, calls) in ranked[:12]:
        print(f"  {op:<16} {t / 1e9:8.3f} s  {100 * t / total_op:5.1f}%  ({calls} chamadas)")
    rest = ranked[12:]
    if rest:
        t = sum(v[0] for _, v in rest)
        print(f"  {'(outros)':<16} {t / 1e9:8.3f} s  {100 * t / total_op:5.1f}%")
    if total_running:
        print(f"\n  tempo dentro de operadores: {total_op / 1e9:.2f} s | "
              f"tempo Running das threads: {total_running / 1e9:.2f} s "
              f"(diferença = sincronização/spin do pool + resto do programa)")

    if by_layer:
        ts = [t for _, t in sorted(by_layer.items())]
        mean = sum(ts) / len(ts)
        print(f"\n  por camada: {len(ts)} camadas, média {mean / 1e9:.3f} s; "
              f"mais cara: camada {max(by_layer, key=by_layer.get)} "
              f"({max(ts) / 1e9:.3f} s), mais barata: camada "
              f"{min(by_layer, key=by_layer.get)} ({min(ts) / 1e9:.3f} s)")


def print_summary(threads, decodes, drops):
    print("\n===== ProfTime: resumo =====")
    for tid, th in sorted(threads.items()):
        totals = {RUNNING: 0, RUNNABLE: 0, IDLE: 0}
        for start, end, state, _cpu in th["slices"]:
            totals[state] += end - start
        window = sum(totals.values())
        if window == 0:
            continue
        pct = {s: 100.0 * t / window for s, t in totals.items()}
        print(f"  {th['comm'] or 'thread':<16} [{tid}]  "
              f"running {pct[RUNNING]:5.1f}%  runnable {pct[RUNNABLE]:5.1f}%  "
              f"idle {pct[IDLE]:5.1f}%  (janela {window / 1e9:.2f} s)")

    if decodes:
        ttft = (decodes[0][1] - decodes[0][0]) / 1e6
        print(f"\n  llama_decode: {len(decodes)} chamadas")
        print(f"  prefill (TTFT): {ttft:.1f} ms")
        if len(decodes) > 1:
            tpot = sum(e - s for s, e in decodes[1:]) / len(decodes[1:]) / 1e6
            print(f"  decode  (TPOT): {tpot:.1f} ms/token  ≈ {1000.0 / tpot:.1f} tokens/s")

    if drops:
        print(f"\n  atenção: {drops} eventos perdidos (ring buffer cheio)")


def main():
    args, workload = parse_args()

    if os.geteuid() != 0:
        sys.exit("erro: programas eBPF exigem root. Rode com sudo.")

    libllama = find_lib(workload, "libllama.so", args.llama_lib)
    if libllama is None:
        print("[proftime] aviso: libllama.so não encontrada; a timeline não terá "
              "a trilha de tokens (use --llama-lib para indicar o caminho)")
    libggml = find_lib(workload, "libggml-cpu.so", args.ggml_lib)
    if libggml is None:
        print("[proftime] aviso: libggml-cpu.so não encontrada; sem rastreio de "
              "operadores (use --ggml-lib para indicar o caminho)")

    # 1) Inicia o workload já PAUSADO (fork + SIGSTOP antes do exec).
    pid = spawn_paused(workload)

    # 2) Compila e carrega o programa eBPF, com o PID alvo como constante.
    print(f"[proftime] compilando e carregando o programa eBPF (alvo: pid {pid})...")
    print("[proftime] (a primeira execução pode levar ~10-30 s: o BCC compila o C com clang)")
    from bcc import BPF
    bpf_src = Path(__file__).with_name("proftime_bpf.c").read_text()
    cflags = [f"-DTARGET_TGID={pid}"]
    if not args.no_cpu_view:
        cflags.append("-DTRACE_ALL_CPUS=1")
    bpf = BPF(text=bpf_src, cflags=cflags)

    # 3) Anexa os uprobes (llama_decode + operadores) e registra a thread principal.
    if libllama:
        bpf.attach_uprobe(name=str(libllama), sym="llama_decode", fn_name="on_decode_begin")
        bpf.attach_uretprobe(name=str(libllama), sym="llama_decode", fn_name="on_decode_end")
    op_syms = attach_op_probes(bpf, libggml) if libggml else []
    if libggml and not op_syms:
        print("[proftime] aviso: libggml-cpu.so sem símbolos (binário stripped?); "
              "compile o llama.cpp da fonte para rastrear operadores (veja o README)")
    bpf["interest"][ctypes.c_uint32(pid)] = ctypes.c_uint8(1)
    print("[proftime] probes anexados (sched_switch, sched_wakeup, fork"
          + (", llama_decode" if libllama else "")
          + (f", {len(op_syms)} operadores ggml)" if op_syms else ")"))

    # 4) Libera o processo e coleta os eventos até ele terminar.
    raw_events, raw_ops = collect_events(bpf, pid)
    drops = bpf["drops"][ctypes.c_int(0)].value

    # 5) Reconstrói a timeline e grava o resultado.
    print(f"\n[proftime] {len(raw_events)} eventos de sched + {len(raw_ops)} de "
          "operadores; gerando timeline...")
    threads, decodes = build_timeline(raw_events)
    cpu_lanes = build_cpu_lanes(raw_events)
    op_spans = defaultdict(list)  # tid -> [(ts, dur, op, tensor)]
    for ts, dur, tid, op, tname in sorted(raw_ops):
        op_spans[tid].append((ts, dur, op, tname))
    write_chrome_trace(args.output, pid, Path(workload[0]).name, threads, decodes,
                       op_spans, cpu_lanes)

    print_summary(threads, decodes, drops)
    print_op_summary(op_spans, threads)
    print(f"\n  trace salvo em: {args.output}")
    print("  visualize em: https://ui.perfetto.dev (Open trace file)\n")

    # Sai direto, sem rodar o destrutor do BCC: desanexar 200+ uprobes um a
    # um é lento (era o "travamento" no final). O kernel remove os probes e
    # libera os mapas sozinho quando o processo termina.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
