#!/usr/bin/env bash
# Baixa o llama.cpp pré-compilado (CPU) e um modelo pequeno para o ProfTime.
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_TAG="b10010"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

if [ ! -x llama.cpp/llama-completion ]; then
    echo "==> baixando llama.cpp ${LLAMA_TAG} (pré-compilado, CPU)..."
    curl -sL -o /tmp/llama-cpu.tar.gz "$LLAMA_URL"
    mkdir -p llama.cpp
    tar xzf /tmp/llama-cpu.tar.gz -C llama.cpp --strip-components=1
    rm /tmp/llama-cpu.tar.gz
fi

if [ ! -f models/qwen2.5-0.5b-instruct-q4_k_m.gguf ]; then
    echo "==> baixando o modelo Qwen2.5-0.5B-Instruct (Q4_K_M, ~470 MB)..."
    mkdir -p models
    curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf "$MODEL_URL"
fi

# Build da fonte, com símbolos: necessária para o rastreio de operadores
# (os binários oficiais são stripped). Exige gcc/g++ e cmake.
if [ ! -x llama.cpp-src/build/bin/llama-completion ]; then
    echo "==> compilando o llama.cpp ${LLAMA_TAG} da fonte (~3 min)..."
    [ -d llama.cpp-src ] || git clone --depth 1 --branch "$LLAMA_TAG" \
        https://github.com/ggml-org/llama.cpp llama.cpp-src
    cmake -S llama.cpp-src -B llama.cpp-src/build \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
    cmake --build llama.cpp-src/build -j"$(nproc)" --target llama-completion
fi

echo "==> pronto. Teste com:"
echo "  sudo python3 proftime.py -o trace.json -- \\"
echo "      ./llama.cpp-src/build/bin/llama-completion \\"
echo "      -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \\"
echo "      -p \"O eBPF é uma tecnologia que\" -n 64 -t 4 --no-warmup --seed 42 -no-cnv"
