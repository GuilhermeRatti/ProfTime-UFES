// proftime_bpf.c — parte do ProfTime que roda DENTRO do kernel Linux.
//
// Este arquivo é compilado em tempo de execução pelo BCC (proftime.py) e
// carregado no kernel como um programa eBPF. Ele NÃO modifica o llama.cpp:
// apenas "escuta" eventos do kernel e de uma função do llama.cpp, e envia
// cada evento para o user space através de um ring buffer.
//
// O que observamos (mesmos pontos do artigo ProfInfer, Tabela 2):
//
//   1. tracepoint sched:sched_switch   -> o escalonador trocou a thread da CPU
//   2. tracepoint sched:sched_wakeup   -> uma thread dormindo foi acordada
//   3. uprobe/uretprobe em llama_decode -> início/fim de um passo de inferência
//                                          (1ª chamada = prefill, demais = 1 token)
//   4. uprobe/uretprobe nas funções ggml_compute_forward_<op> do backend de
//      CPU (libggml-cpu.so) -> início/fim de cada OPERADOR (MUL_MAT, SOFT_MAX,
//      RMS_NORM, ...) em cada thread. É isso que preenche as barras da
//      timeline com "o que" a CPU está computando.
//
// TARGET_TGID é o PID do processo de inferência, definido via flag de
// compilação (-DTARGET_TGID=...) pelo proftime.py. Filtrar por processo é
// essencial: sched_switch dispara milhares de vezes por segundo no sistema
// inteiro, e só nos interessam as threads da inferência.

#include <uapi/linux/ptrace.h>

// Tipos de evento enviados ao user space.
enum ev_kind {
    EV_SWITCH_IN    = 0,  // thread ganhou uma CPU (estado: Running)
    EV_SWITCH_OUT   = 1,  // thread perdeu a CPU (prev_state diz o motivo)
    EV_WAKEUP       = 2,  // thread acordou e está pronta (estado: Runnable)
    EV_DECODE_BEGIN = 3,  // llama_decode() foi chamada
    EV_DECODE_END   = 4,  // llama_decode() retornou
    EV_CPU          = 5,  // QUALQUER thread (de qualquer processo) entrou na
                          // CPU — alimenta a visão de ocupação por núcleo
};

// Um evento. Struct enxuta: cada campo custa espaço no ring buffer.
struct event_t {
    u64 ts;          // timestamp em ns (CLOCK_MONOTONIC)
    s64 prev_state;  // só em EV_SWITCH_OUT: 0 = perdeu a CPU mas quer rodar
                     // (preempção -> Runnable); != 0 = bloqueou/dormiu (Idle)
    u32 tid;         // id da thread (o que o kernel chama de pid)
    u32 cpu;         // em qual CPU o evento ocorreu
    u32 kind;        // um dos enum ev_kind
    char comm[16];   // nome da thread
};

// Canal kernel -> user space: ring buffer de 512 páginas (2 MiB).
BPF_RINGBUF_OUTPUT(events, 512);

// Conjunto de threads que pertencem à inferência (tid -> 1).
// O proftime.py insere a thread principal; o handler de fork abaixo
// adiciona automaticamente cada thread nova criada pelo processo.
BPF_HASH(interest, u32, u8);

// Contador de eventos perdidos por ring buffer cheio (trade-off citado
// no artigo: ring buffer é eficiente, mas pode descartar eventos).
BPF_ARRAY(drops, u64, 1);

// ---------------------------------------------------------------------------
// Rastreio de OPERADORES (nível mais fino, seção 3.3.3 do artigo).
//
// Espelho mínimo do struct ggml_tensor (ggml/include/ggml.h, llama.cpp
// b10010) — o eBPF precisa do layout exato para ler os campos `op` e `name`
// da memória do processo. Só os campos até `name` importam aqui.
// ---------------------------------------------------------------------------
#define GGML_MAX_DIMS      4
#define GGML_MAX_OP_PARAMS 64
#define GGML_MAX_SRC       10
#define GGML_MAX_NAME      64

struct ggml_tensor {
    int  type;                                  // enum ggml_type
    void *buffer;
    long long ne[GGML_MAX_DIMS];                // dimensões
    unsigned long nb[GGML_MAX_DIMS];            // strides
    int  op;                                    // enum ggml_op  <- queremos
    int  op_params[GGML_MAX_OP_PARAMS / 4];
    int  flags;
    void *src[GGML_MAX_SRC];
    void *view_src;
    unsigned long view_offs;
    void *data;
    char name[GGML_MAX_NAME];                   // ex.: "ffn_out-12" <- queremos
    void *extra;
    char padding[8];
};

// Evento de operador: um por chamada ggml_compute_forward_<op> por thread.
struct op_event_t {
    u64 ts;                    // início (ns)
    u64 dur;                   // duração (ns)
    u32 tid;
    u32 op;                    // enum ggml_op (nome resolvido no user space)
    char tname[GGML_MAX_NAME]; // nome do tensor de saída (revela a camada)
};

// Canal separado dos eventos de sched: operadores disparam com MUITO mais
// frequência (~milhares por token), então ganham um ring buffer maior (8 MiB).
BPF_RINGBUF_OUTPUT(op_events, 2048);

// Estado pendente entre a entrada e o retorno da função, por thread.
struct op_stash_t {
    u64 ts;
    u32 op;
    char tname[GGML_MAX_NAME];
};
BPF_HASH(op_begin, u32, struct op_stash_t);

// uprobe: entrada de ggml_compute_forward_<op>(params, tensor).
// Lê `op` e `name` do 2º argumento e guarda com o timestamp, por thread.
int on_op_begin(struct pt_regs *ctx)
{
    u64 id = bpf_get_current_pid_tgid();
    if ((id >> 32) != TARGET_TGID)
        return 0;

    struct ggml_tensor *t = (struct ggml_tensor *)PT_REGS_PARM2(ctx);
    struct op_stash_t st = {};
    st.ts = bpf_ktime_get_ns();
    bpf_probe_read_user(&st.op, sizeof(st.op), &t->op);
    bpf_probe_read_user(&st.tname, sizeof(st.tname), &t->name);

    u32 tid = (u32)id;
    op_begin.update(&tid, &st);
    return 0;
}

// uretprobe: retorno da mesma função. Calcula a duração e emite o evento.
int on_op_end(struct pt_regs *ctx)
{
    u64 id = bpf_get_current_pid_tgid();
    if ((id >> 32) != TARGET_TGID)
        return 0;

    u32 tid = (u32)id;
    struct op_stash_t *st = op_begin.lookup(&tid);
    if (!st)
        return 0;

    struct op_event_t ev = {};
    ev.ts = st->ts;
    ev.dur = bpf_ktime_get_ns() - st->ts;
    ev.tid = tid;
    ev.op = st->op;
    __builtin_memcpy(&ev.tname, st->tname, sizeof(ev.tname));
    op_begin.delete(&tid);

    if (op_events.ringbuf_output(&ev, sizeof(ev), 0) != 0) {
        int zero = 0;
        u64 *d = drops.lookup(&zero);
        if (d)
            __sync_fetch_and_add(d, 1);
    }
    return 0;
}

static __always_inline void submit(u32 kind, u32 tid, u32 cpu,
                                   s64 prev_state, const void *comm_src)
{
    struct event_t ev = {};
    ev.ts = bpf_ktime_get_ns();
    ev.kind = kind;
    ev.tid = tid;
    ev.cpu = cpu;
    ev.prev_state = prev_state;
    bpf_probe_read_kernel(&ev.comm, sizeof(ev.comm), comm_src);

    if (events.ringbuf_output(&ev, sizeof(ev), 0) != 0) {
        int zero = 0;
        u64 *d = drops.lookup(&zero);
        if (d)
            __sync_fetch_and_add(d, 1);
    }
}

// ---------------------------------------------------------------------------
// 1) Rastreio de criação de threads: quando o processo alvo cria uma thread
//    nova (o pool de threads do ggml, por exemplo), registramos o tid dela.
// ---------------------------------------------------------------------------
TRACEPOINT_PROBE(sched, sched_process_fork)
{
    u32 parent_tgid = bpf_get_current_pid_tgid() >> 32;
    if (parent_tgid != TARGET_TGID)
        return 0;

    u32 child_tid = args->child_pid;
    u8 one = 1;
    interest.update(&child_tid, &one);
    return 0;
}

// ---------------------------------------------------------------------------
// 2) Troca de contexto: dispara TODA vez que o escalonador troca a thread
//    de uma CPU. "prev" é quem saiu, "next" é quem entrou.
// ---------------------------------------------------------------------------
TRACEPOINT_PROBE(sched, sched_switch)
{
    u32 cpu = bpf_get_smp_processor_id();

    u32 prev_tid = args->prev_pid;
    if (interest.lookup(&prev_tid))
        submit(EV_SWITCH_OUT, prev_tid, cpu, args->prev_state, args->prev_comm);

    u32 next_tid = args->next_pid;
    if (interest.lookup(&next_tid))
        submit(EV_SWITCH_IN, next_tid, cpu, 0, args->next_comm);

#ifdef TRACE_ALL_CPUS
    // Visão de ocupação por núcleo: registra QUEM entrou em cada CPU, de
    // qualquer processo (tid 0 = "swapper", a thread ociosa do kernel).
    // É o que permite ver interferência externa, como na Figura 8 do artigo.
    submit(EV_CPU, next_tid, cpu, 0, args->next_comm);
#endif

    return 0;
}

// ---------------------------------------------------------------------------
// 3) Despertar de threads: uma thread que dormia (Idle) ficou pronta para
//    rodar (Runnable). Ela só volta a executar quando o sched_switch a
//    colocar numa CPU — o intervalo entre os dois é tempo de espera por CPU.
// ---------------------------------------------------------------------------
TRACEPOINT_PROBE(sched, sched_wakeup)
{
    u32 tid = args->pid;
    if (interest.lookup(&tid))
        submit(EV_WAKEUP, tid, bpf_get_smp_processor_id(), 0, args->comm);
    return 0;
}

// Mesmo evento, mas para threads recém-criadas (primeiro despertar).
TRACEPOINT_PROBE(sched, sched_wakeup_new)
{
    u32 tid = args->pid;
    if (interest.lookup(&tid))
        submit(EV_WAKEUP, tid, bpf_get_smp_processor_id(), 0, args->comm);
    return 0;
}

// ---------------------------------------------------------------------------
// 4) uprobe/uretprobe em llama_decode (libllama.so): marca o início e o fim
//    de cada passo de inferência, sem recompilar o llama.cpp. A diferença
//    entre os timestamps dá o TTFT (1ª chamada) e o TPOT (demais chamadas).
// ---------------------------------------------------------------------------
int on_decode_begin(struct pt_regs *ctx)
{
    u64 id = bpf_get_current_pid_tgid();
    if ((id >> 32) != TARGET_TGID)
        return 0;

    char comm[16];
    bpf_get_current_comm(&comm, sizeof(comm));
    submit(EV_DECODE_BEGIN, (u32)id, bpf_get_smp_processor_id(), 0, comm);
    return 0;
}

int on_decode_end(struct pt_regs *ctx)
{
    u64 id = bpf_get_current_pid_tgid();
    if ((id >> 32) != TARGET_TGID)
        return 0;

    char comm[16];
    bpf_get_current_comm(&comm, sizeof(comm));
    submit(EV_DECODE_END, (u32)id, bpf_get_smp_processor_id(), 0, comm);
    return 0;
}
