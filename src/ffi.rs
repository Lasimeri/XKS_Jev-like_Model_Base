//! The C shim's functions (`csrc/shim.c`): scalars and pointers only, every
//! llama.cpp structure is built on the C side against llama.h.

use std::ffi::c_char;

/// llama.cpp's `struct llama_model`, only ever behind a pointer.
#[repr(C)]
pub struct LlamaModel {
    _opaque: [u8; 0],
}

/// llama.cpp's `struct llama_context`, only ever behind a pointer.
#[repr(C)]
pub struct LlamaContext {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    pub fn jl_init(verbose: i32);
    pub fn jl_system_info() -> *const c_char;

    pub fn jl_model_load(path: *const c_char, load_mode: i32, repack: i32) -> *mut LlamaModel;
    pub fn jl_model_free(m: *mut LlamaModel);
    pub fn jl_model_desc(m: *const LlamaModel, buf: *mut c_char, len: usize) -> i32;
    pub fn jl_model_size(m: *const LlamaModel) -> u64;
    pub fn jl_model_n_params(m: *const LlamaModel) -> u64;
    pub fn jl_model_recurrent(m: *const LlamaModel) -> i32;
    pub fn jl_n_vocab(m: *const LlamaModel) -> i32;
    pub fn jl_bos(m: *const LlamaModel) -> i32;
    pub fn jl_tokenize(
        m: *const LlamaModel,
        text: *const c_char,
        len: i32,
        add_special: i32,
        parse_special: i32,
        out: *mut i32,
        cap: i32,
    ) -> i32;

    pub fn jl_context_new(
        m: *mut LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_ubatch: u32,
        n_seq_max: u32,
        n_outputs_max: u32,
        n_threads: i32,
        flash_attn: i32,
    ) -> *mut LlamaContext;
    pub fn jl_context_free(c: *mut LlamaContext);
    pub fn jl_decode(
        c: *mut LlamaContext,
        n: i32,
        tok: *const i32,
        pos: *const i32,
        seq: *const i32,
        out: *const i8,
    ) -> i32;
    pub fn jl_logits(c: *mut LlamaContext, i: i32) -> *const f32;
    pub fn jl_mem_clear(c: *mut LlamaContext);
    pub fn jl_seq_rm(c: *mut LlamaContext, seq: i32, p0: i32, p1: i32) -> i32;
    pub fn jl_seq_cp(c: *mut LlamaContext, src: i32, dst: i32, p0: i32, p1: i32);
}
