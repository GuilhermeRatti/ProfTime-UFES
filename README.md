# ProfTime — timeline de threads de inferência LLM com eBPF

Reimplementação didática e mínima da visão **ProfTime** do artigo
[ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler](paper/2601.20755v2.pdf)
(arXiv:2601.20755). Produzido por Arthur Roberto Barbosa Maciel e Guilherme Ratti Moraes. A ideia: observar, **sem modificar nem recompilar** o motor
de inferência (llama.cpp), o que as threads da inferência estão fazendo a cada
instante — executando, esperando por CPU ou dormindo — e em qual CPU, junto com
as fronteiras de cada passo de geração de token.

O resultado é uma timeline interativa (aberta no [Perfetto](https://ui.perfetto.dev))
e um resumo textual com TTFT, TPOT e a ocupação de cada thread.

## Contexto em 1 minuto

**Inferência de LLM**: um modelo como o Qwen gera texto token por token. A
primeira chamada processa o prompt inteiro (*prefill*, medida pelo TTFT — time
to first token); cada chamada seguinte gera um token (*decode*, medida pelo
TPOT — time per output token). No llama.cpp, cada um desses passos é uma chamada
à função `llama_decode`, e o trabalho pesado (multiplicações de matrizes) é
dividido entre um pool de threads.

**eBPF**: tecnologia do kernel Linux que permite executar pequenos programas
verificados *dentro do kernel*, anexados a eventos — sem alterar o kernel nem a
aplicação observada. Dois tipos de anexo nos interessam:

- **tracepoints**: pontos de instrumentação estáveis do próprio kernel
  (ex.: `sched_switch` dispara a cada troca de contexto do escalonador);
- **uprobes**: breakpoints dinâmicos em funções de *user space*
  (ex.: na função `llama_decode` da `libllama.so`).

## Como funciona

```
                     USER SPACE                          KERNEL SPACE
  ┌───────────────────────────────────┐    ┌────────────────────────────────────┐
  │ proftime.py                       │    │ proftime_bpf.c (programa eBPF)     │
  │                                   │    │                                    │
  │ 1. inicia llama.cpp PAUSADO       │    │  tracepoint sched_process_fork ────┼─ registra threads novas
  │ 2. compila/carrega o BPF (BCC) ───┼──▶ │  tracepoint sched_switch ──────────┼─ quem entrou/saiu da CPU
  │ 3. registra o PID no mapa         │    │  tracepoint sched_wakeup ──────────┼─ quem acordou
  │ 4. libera o processo (SIGCONT)    │    │  uprobe/uretprobe llama_decode ────┼─ início/fim de cada passo
  │ 5. lê eventos dos ring buffers ◀──┼────┼── uprobe/uretprobe ────────────────┼─ cada OPERADOR (MUL_MAT,
  │ 6. reconstrói estados e gera      │    │   ggml_compute_forward_<op>        │  SOFT_MAX...) por thread
  │    o JSON (Chrome Trace Format)   │    │                                    │
  │                                   │    │  filtro: só threads da inferência  │
  └───────────────────────────────────┘    └────────────────────────────────────┘
```

O processo de inferência nasce **pausado** (SIGSTOP antes do `exec`): assim os
probes já estão ativos quando o llama.cpp cria suas threads de trabalho, e
nenhum evento é perdido no início.

### A máquina de estados das threads

Cada thread da inferência está sempre em um de três estados (artigo, seção 4.2):

| Estado | Significado | Como detectamos |
|---|---|---|
| **Running** | executando em alguma CPU | `sched_switch` colocou a thread na CPU |
| **Runnable** | pronta, mas esperando CPU livre | `sched_wakeup`, ou `sched_switch` tirou a thread com `prev_state == 0` (preempção) |
| **Idle** | dormindo (ex.: esperando trabalho num futex) | `sched_switch` tirou a thread com `prev_state != 0` (bloqueio voluntário) |

O tempo em *Runnable* é especialmente interessante: é tempo em que a thread
**queria** rodar mas o escalonador não deu CPU — sinal de interferência de
outras cargas (Figura 8 do artigo).

### Rastreio de operadores: o que preenche as barras

Saber que uma thread está *Running* não diz **no que** ela gasta a CPU. Para
isso (seção 3.3.3 do artigo), anexamos uprobes às funções
`ggml_compute_forward_<op>` do backend de CPU (`libggml-cpu.so`) — uma por
operador do grafo: `mul_mat`, `soft_max`, `rms_norm`, `rope`, ... Todas têm a
mesma assinatura `(params, tensor)`, então um único par de handlers serve para
todas. O handler lê, da memória do processo, dois campos do `struct
ggml_tensor` (espelhado em [proftime_bpf.c](proftime_bpf.c)):

- `op` — o tipo do operador (enum, traduzido pela tabela [ggml_ops.py](ggml_ops.py));
- `name` — o nome do tensor de saída, ex.: `ffn_out-12` → camada 12.

Com isso o resumo final responde a pergunta de ouro — *onde está o gargalo?* —
com o tempo de CPU somado por operador e por camada, e a timeline mostra cada
operador colorido dentro da barra de cada thread.

**Requisito**: os binários oficiais do llama.cpp são *stripped* (sem tabela de
símbolos), e essas funções não são exportadas — então o rastreio de operadores
exige compilar o llama.cpp da fonte (2–3 min; o `setup.sh` já faz isso):

```bash
git clone --depth 1 --branch b10010 https://github.com/ggml-org/llama.cpp llama.cpp-src
cmake -S llama.cpp-src -B llama.cpp-src/build \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp-src/build -j$(nproc) --target llama-completion
```

e usar `./llama.cpp-src/build/bin/llama-completion` como workload. (Um detalhe
de compilador: o *dispatcher* `ggml_compute_forward` é `static` com um único
ponto de chamada e é inlined pelo GCC — por isso sondamos as funções por
operador, que são externas e sobrevivem à otimização.)

## Arquivos

| Arquivo | Papel |
|---|---|
| [proftime_bpf.c](proftime_bpf.c) | programa eBPF (roda no kernel): captura os eventos e os envia pelo ring buffer |
| [proftime.py](proftime.py) | user space: carrega o BPF via BCC, lança o workload, reconstrói a timeline e gera o JSON |
| [proftime_plot.py](proftime_plot.py) | gera figuras estáticas (PNG) — timeline, gargalo por operador e ocupação por núcleo |
| [ggml_ops.py](ggml_ops.py) | tabela enum ggml_op → nome, gerada do ggml.h (b10010) |
| [setup.sh](setup.sh) | baixa llama.cpp pré-compilado + modelo (Qwen2.5-0.5B) e compila o llama.cpp da fonte |
| [figures/](figures/) | figuras geradas a partir do trace (usadas no relatório) |
| [paper/](paper/) | o PDF do artigo ProfInfer |

Os diretórios pesados e regeneráveis (`llama.cpp/`, `llama.cpp-src/`, `models/`)
e o `trace.json` (~30 MB por execução) ficam fora do controle de versão
(veja o [.gitignore](.gitignore)); o `setup.sh` reconstrói tudo.

## Como rodar

Requisitos: Linux com BCC instalado (`sudo apt install python3-bpfcc`) e root
(todo programa eBPF exige privilégios para ser carregado no kernel).

```bash
./setup.sh   # baixa llama.cpp (16 MB) e o modelo Qwen2.5-0.5B (~470 MB)

sudo python3 proftime.py -o trace.json -- \
    ./llama.cpp-src/build/bin/llama-completion \
    -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -p "O eBPF é uma tecnologia que" -n 64 -t 4 --no-warmup --seed 42 -no-cnv
```

(Use `./llama.cpp/llama-completion` — o binário pré-compilado — se não tiver
feito a build da fonte; tudo funciona igual, só sem a trilha de operadores.)

O `-no-cnv` é importante: sem ele, o llama.cpp detecta que o modelo tem
template de chat e entra em **modo conversa interativo** (fica esperando
input no terminal em vez de gerar uma vez e sair).

Ao final, o script imprime um resumo (ocupação por thread, TTFT, TPOT) e grava
o `trace.json`.

## Visualizando a timeline

**Interativa (como no artigo)**: abra <https://ui.perfetto.dev> → *Open trace
file* → selecione o `trace.json`. As figuras de ProfTime do artigo são
exatamente essa visão do Perfetto. Navegação: `W`/`S` dão zoom, `A`/`D` movem,
e clicar numa fatia mostra os detalhes. Cada linha é uma thread; a trilha
*tokens (llama_decode)* mostra as fronteiras do prefill e de cada token gerado.

**Estática (para o relatório)**: `proftime_plot.py` desenha a mesma timeline
com matplotlib (não precisa de root; só lê o JSON):

```bash
python3 proftime_plot.py trace.json -o proftime_full.png            # trace inteiro
python3 proftime_plot.py trace.json --start 1.8 --end 2.1 -o zoom.png  # janela em segundos
python3 proftime_plot.py trace.json --op-chart gargalo.png          # tempo por operador
python3 proftime_plot.py trace.json --cpus nucleos.png              # ocupação por núcleo
```

A visão `--cpus` é a da Figura 8 do artigo: uma linha por núcleo, mostrando
qual thread ocupa cada um ao longo do tempo — threads da inferência coloridas,
outros processos em cinza, vazio = núcleo ocioso. Para isso o `proftime.py`
registra, por padrão, quem entra em cada CPU a cada troca de contexto (de
qualquer processo; desligue com `--no-cpu-view`). No Perfetto, essas trilhas
aparecem no grupo "CPUs (ocupação por núcleo)".

Estados são codificados por cor **e** altura da barra (Running alta e verde,
Runnable média e âmbar, Idle um filete cinza), então a figura continua legível
em impressão P&B e para daltônicos. Quando o trace tem operadores, eles são
desenhados por cima do estado Running — o verde que sobrar aparecendo é CPU
ocupada **sem** operador (spin/sincronização do pool de threads).

## O que observar na timeline

- **Busy-waiting do pool de threads**: com a configuração padrão, os workers
  aparecem *Running* contínuos durante toda a rajada de decode — o pool do
  ggml fica "girando" na CPU à espera de trabalho (`--poll 50`, padrão) em vez
  de dormir. Rode o workload com `--poll 0` e compare: os workers passam a
  alternar *Running* / *Idle* a cada operador (dormem num futex e são
  acordados), como nas Figuras 5–6 do artigo. É um trade-off clássico:
  latência menor (spin) vs. CPU/energia menor (sleep) — visível só com um
  profiler assim.
- **Interferência**: rode algo pesado em paralelo (ex.: `stress-ng -c 16`) e
  veja fatias *Runnable* aparecerem — a inferência pronta para rodar, mas sem
  CPU (Figura 8 do artigo).
- **Prefill vs. decode**: o primeiro `llama_decode` (prefill) é bem mais longo
  que os demais (um por token).
- **Desbalanceamento**: threads que passam boa parte de um operador em *Idle*
  enquanto outra trabalha (Figura 6a do artigo; visível com `--poll 0`).

## Diferenças em relação ao ProfInfer completo

Implementamos a visão ProfTime nos níveis de thread, token e operador (CPU). O
ProfInfer completo também rastreia grafos e backends de GPU/NPU (uprobe em
`ggml_backend_graph_compute_async`, etc.), lê contadores de hardware (PMCs) por
operador, extrai dimensões dos tensores, gera as visões ProfDAG/ProfStat e
desliga probes dinamicamente conforme um requisito de QoS. A engenharia é a
mesma vista aqui — mais probes, mais dados por evento.

Sobre **overhead**: cada operador custa um uprobe + uretprobe (~1–2 µs cada).
Num modelo pequeno como o Qwen2.5-0.5B, em que um operador médio dura poucas
dezenas de µs, isso infla o TPOT de forma perceptível — compare o `eval time`
com e sem o profiler para medir; num modelo maior o efeito relativo cai. O
artigo ataca exatamente isso com o controle de QoS que desliga probes quando a
inferência fica lenta demais.

**Nota técnica**: o texto do artigo descreve `prev_state == 0` como transição
para *idle* e `prev_state == 1` como *runnable*; na semântica do kernel Linux,
`prev_state == 0` (`TASK_RUNNING`) significa que a thread foi *preemptada* (e
continua pronta para rodar), enquanto valores diferentes de zero indicam
bloqueio voluntário. Seguimos a semântica do kernel, que é também a usada pelo
Perfetto.
