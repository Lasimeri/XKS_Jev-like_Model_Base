/* shim.c: the flat C surface of llama.cpp that XKS_Jev-like_Model_Base calls.
 *
 * Every structure llama.cpp passes by value (model parameters, context
 * parameters, the batch) is built here, against llama.h itself, so the Rust
 * side declares only scalars and pointers and never copies a struct layout
 * by hand. Each function is a thin pass-through: no policy lives here.
 *
 * The batch is allocated per decode exactly as Intel-Phi-Jev's ARTICHOKE
 * engine does (llama_batch_init(n, 0, 1), one sequence id per token), so
 * the two engines hand llama.cpp identical decodes. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "llama.h"

static int jl_verbose;
static enum ggml_log_level jl_last_level = GGML_LOG_LEVEL_INFO;

/* Warnings and errors always; llama.cpp's informational log only when asked.
 * A continuation line inherits the level of the line it continues. */
static void jl_log(enum ggml_log_level level, const char *text, void *user) {
    (void)user;
    if (level == GGML_LOG_LEVEL_CONT) {
        level = jl_last_level;
    } else {
        jl_last_level = level;
    }
    if (text != NULL && (level >= GGML_LOG_LEVEL_WARN || jl_verbose)) {
        fputs(text, stderr);
    }
}

void jl_init(int verbose) {
    jl_verbose = verbose;
    llama_log_set(jl_log, NULL);
    llama_backend_init();
}

/* llama.cpp's own report of the CPU features its kernels were built with
 * and found ("AVX2 = 1 | FMA = 1 | ..."). */
const char *jl_system_info(void) {
    return llama_print_system_info();
}

/* load_mode: 0 = read into process memory, 1 = memory-map (llama_load_mode).
 * repack: let the CPU backend repack weights into its interleaved layouts. */
struct llama_model *jl_model_load(const char *path, int32_t load_mode, int32_t repack) {
    struct llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = 0;
    p.load_mode = (enum llama_load_mode)load_mode;
    p.use_extra_bufts = repack != 0;
    return llama_model_load_from_file(path, p);
}

void jl_model_free(struct llama_model *m) {
    llama_model_free(m);
}

int32_t jl_model_desc(const struct llama_model *m, char *buf, size_t len) {
    return llama_model_desc(m, buf, len);
}

uint64_t jl_model_size(const struct llama_model *m) {
    return llama_model_size(m);
}

uint64_t jl_model_n_params(const struct llama_model *m) {
    return llama_model_n_params(m);
}

/* Recurrent state anywhere in the model (pure recurrent or hybrid): such a
 * state cannot be cut back to an earlier position. */
int32_t jl_model_recurrent(const struct llama_model *m) {
    return llama_model_is_recurrent(m) || llama_model_is_hybrid(m);
}

int32_t jl_n_vocab(const struct llama_model *m) {
    return llama_vocab_n_tokens(llama_model_get_vocab(m));
}

int32_t jl_bos(const struct llama_model *m) {
    return llama_vocab_bos(llama_model_get_vocab(m));
}

/* Returns the token count, or minus the count needed when cap is too small. */
int32_t jl_tokenize(const struct llama_model *m, const char *text, int32_t len,
                    int32_t add_special, int32_t parse_special, int32_t *out, int32_t cap) {
    return llama_tokenize(llama_model_get_vocab(m), text, len, out, cap,
                          add_special != 0, parse_special != 0);
}

/* flash_attn: -1 auto, 0 off, 1 on (llama_flash_attn_type). The cache is
 * unified: forks share the session's attention cells instead of copying
 * them into per-sequence streams. */
struct llama_context *jl_context_new(struct llama_model *m, uint32_t n_ctx, uint32_t n_batch,
                                     uint32_t n_ubatch, uint32_t n_seq_max,
                                     uint32_t n_outputs_max, int32_t n_threads,
                                     int32_t flash_attn) {
    struct llama_context_params p = llama_context_default_params();
    p.n_ctx = n_ctx;
    p.n_batch = n_batch;
    p.n_ubatch = n_ubatch < n_batch ? n_ubatch : n_batch;
    p.n_seq_max = n_seq_max;
    p.n_outputs_max = n_outputs_max;
    p.n_threads = n_threads;
    p.n_threads_batch = n_threads;
    p.kv_unified = true;
    p.no_perf = true;
    p.flash_attn_type = (enum llama_flash_attn_type)flash_attn;
    return llama_init_from_model(m, p);
}

void jl_context_free(struct llama_context *c) {
    llama_free(c);
}

/* One llama_decode of n tokens: token, position, sequence, output flag. */
int32_t jl_decode(struct llama_context *c, int32_t n, const int32_t *tok, const int32_t *pos,
                  const int32_t *seq, const int8_t *out) {
    struct llama_batch b = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; i++) {
        b.token[i] = tok[i];
        b.pos[i] = pos[i];
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq[i];
        b.logits[i] = out[i];
    }
    b.n_tokens = n;
    const int32_t rc = llama_decode(c, b);
    llama_batch_free(b);
    return rc;
}

/* The n_vocab logits of output row i of the last decode. */
const float *jl_logits(struct llama_context *c, int32_t i) {
    return llama_get_logits_ith(c, i);
}

void jl_mem_clear(struct llama_context *c) {
    llama_memory_clear(llama_get_memory(c), true);
}

int32_t jl_seq_rm(struct llama_context *c, int32_t seq, int32_t p0, int32_t p1) {
    return llama_memory_seq_rm(llama_get_memory(c), seq, p0, p1);
}

void jl_seq_cp(struct llama_context *c, int32_t src, int32_t dst, int32_t p0, int32_t p1) {
    llama_memory_seq_cp(llama_get_memory(c), src, dst, p0, p1);
}
