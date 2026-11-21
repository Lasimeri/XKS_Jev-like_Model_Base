# XKS_Jev-like_Model_Base
#
#   make               fetch + build llama.cpp for this CPU, then jevbase
#   make run           the terminal interface
#   make serve         POST /v1/systemone on 127.0.0.1:8097
#   make bench         decisions per second (cold, warm, single)
#   make check         unit tests + clippy (no model needed)
#   make parity XKS=/path/to/xks
#                      the oracle gate: Intel-Phi-Jev's xks and jevbase on the
#                      same 32 requests, outputs compared
#   make install       copy jevbase to $(PREFIX)/bin (default ~/.local/bin)
#
# llama.cpp is pinned and built for x86-64 CPUs with AVX2 + FMA (+ F16C,
# BMI2): its matmul, quantized dot-product and attention kernels take their
# AVX2/FMA paths, OpenMP threads, one static CPU backend linked into the
# binary. Builds use at most JOBS (8) compile jobs.
#
# No path of the machine that builds it goes into the binary: llama.cpp's
# __FILE__ strings (its asserts) and rustc's panic locations are rewritten to
# paths relative to their source tree, so two checkouts of the same commit in
# different folders build byte-identical binaries.

LLAMA_REPO   ?= https://github.com/ggml-org/llama.cpp
LLAMA_COMMIT ?= f5b9bd39b56c7a7839a9795a100b6a00b84ac961
LLAMA_DIR    := third_party/llama.cpp
JOBS         ?= 8
PREFIX       ?= $(HOME)/.local
XKS          ?= ../Intel-Phi-Jev/target/release/xks
BIN          := target/release/jevbase
CARGO_HOME   ?= $(HOME)/.cargo

PATH_MAP     := -ffile-prefix-map=$(abspath $(LLAMA_DIR))=llama.cpp
RUST_REMAP   := --remap-path-prefix=$(CARGO_HOME)=cargo --remap-path-prefix=$(CURDIR)=.

LLAMA_FLAGS := -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_C_FLAGS="$(PATH_MAP)" -DCMAKE_CXX_FLAGS="$(PATH_MAP)" \
	-DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON \
	-DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
	-DGGML_AVX512=OFF -DGGML_AVX_VNNI=OFF \
	-DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BACKEND_DL=OFF -DGGML_OPENMP=ON \
	-DGGML_CCACHE=OFF \
	-DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
	-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
	-DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF

SHIM_FLAGS := -std=c11 -O2 -fPIC -Wall -Wextra -Werror \
	-I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include

.PHONY: all build llama llama-shared run serve bench check parity install

all: build

build: llama compile_commands.json
	RUSTFLAGS="$$RUSTFLAGS $(RUST_REMAP)" cargo build --release -j $(JOBS)

llama: $(LLAMA_DIR)/build-cpu/src/libllama.a

# Shared libraries of the same source and flags, for building the oracle
# (Intel-Phi-Jev's xks: LLAMA_CPP_DIR and LLAMA_BUILD_DIR, see README).
llama-shared: $(LLAMA_DIR)/build-shared/bin/libllama.so

$(LLAMA_DIR)/.git/HEAD:
	git init -q $(LLAMA_DIR)
	git -C $(LLAMA_DIR) fetch -q --depth 1 $(LLAMA_REPO) $(LLAMA_COMMIT)
	git -C $(LLAMA_DIR) checkout -q FETCH_HEAD
	test "$$(git -C $(LLAMA_DIR) rev-parse HEAD)" = "$(LLAMA_COMMIT)"

$(LLAMA_DIR)/build-cpu/src/libllama.a: Makefile | $(LLAMA_DIR)/.git/HEAD
	cmake -S $(LLAMA_DIR) -B $(LLAMA_DIR)/build-cpu $(LLAMA_FLAGS) \
		-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	cmake --build $(LLAMA_DIR)/build-cpu -j $(JOBS) --target llama
	touch $@

$(LLAMA_DIR)/build-shared/bin/libllama.so: Makefile | $(LLAMA_DIR)/.git/HEAD
	cmake -S $(LLAMA_DIR) -B $(LLAMA_DIR)/build-shared $(LLAMA_FLAGS) -DBUILD_SHARED_LIBS=ON
	cmake --build $(LLAMA_DIR)/build-shared -j $(JOBS) --target llama
	touch $@

# For clangd: the shim's real compile command (build.rs uses the same flags).
compile_commands.json: Makefile
	printf '[{"directory":"%s","file":"csrc/shim.c","command":"cc %s -c csrc/shim.c"}]\n' \
		"$(CURDIR)" "$(SHIM_FLAGS)" > $@

run: build
	$(BIN)

serve: build
	$(BIN) serve

bench: build
	$(BIN) bench

check: llama
	cargo test --release -j $(JOBS)
	cargo clippy --release -j $(JOBS) --all-targets -- -D warnings

# Both engines on the same requests, in the same order, from a fresh start:
# exact readout and the oracle's x86 settings (8 threads, repack off). The
# outputs must be identical but for the model name and latencies.
parity: build
	@test -x "$(XKS)" || { echo "XKS=$(XKS) is not an executable xks (see README, Oracle)"; exit 1; }
	mkdir -p out/parity
	env XKS_SYSTEM_FILE= XKS_GPU_LAYERS= XKS_SITE=x86 XKS_KILL_DATE=0 XKS_LAYOUT=letters \
		XKS_PERMUTATIONS=3 tests/parity/serve-and-ask.sh 8091 tests/parity/requests.jsonl \
		out/parity/oracle-letters.jsonl -- $(XKS) --site x86 --threads 8 --debug serve --bind 127.0.0.1:8091
	tests/parity/serve-and-ask.sh 8092 tests/parity/requests.jsonl out/parity/jevbase-letters.jsonl -- \
		$(BIN) --debug --set readout=exact --set repack=false -t 8 --set bind=127.0.0.1:8092 serve
	jq -n --slurpfile a out/parity/oracle-letters.jsonl --slurpfile b out/parity/jevbase-letters.jsonl \
		-f tests/parity/compare.jq

install: build
	install -Dm755 $(BIN) $(PREFIX)/bin/jevbase
