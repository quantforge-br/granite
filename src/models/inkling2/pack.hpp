// src/models/inkling2/pack.hpp - the Inkling 2 model file as the loader sees it: geometry and loop from the meta's
// gguf_kv, the tensor table, the experts layout, and where the vision tower and the drafter sit.  The file is a .granite
// pack (tools/convert/checkpoint_inkling2.py writes it) or llama.cpp's inkling2_loop GGUF, whose names, shapes and
// expert tensors open_pack maps onto the pack's.
#pragma once

#include "granite/core/loader.hpp"
#include "granite/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace granite::models::inkling2 {

inline constexpr const char* kArch = "inkling2";
/// llama.cpp's name for the model: its GGUFs' general.architecture and key prefix.
inline constexpr const char* kGgufArch = "inkling2_loop";

/// One application of a layer: the layer, its loop pass (0 outside the loop), and its own cache slot.
struct Visit {
    int layer = 0, pass = 0, slot = 0;
};

struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    int window = 0, rel = 0, extent = 0, conv = 0, block = 0;
    float qk_clip = 0.f, rel_clip = 0.f, log_floor = 0.f, log_alpha = 0.f, rms_eps = 1e-6f;
    int n_expert = 0, n_used = 0, n_ff_exp = 0;
    float expert_scale = 1.f, gate_softcap = 0.f, up_softcap = 0.f, logit_softcap = 0.f;
    int loop_start = 0, loop_end = 0, loop_passes = 1, gate_rank = 0;
    int image_token = -1;
    int64_t context_length = 0;
    std::vector<bool> swa;           ///< per layer: a sliding-window layer
    std::vector<Visit> visits;       ///< the layer applications of one token, in order
    bool looped(int layer) const { return layer >= loop_start && layer < loop_end; }
    bool boundary(int layer) const { return layer % block == 0; }
};

/// The DSpark drafter's settings (`inkling2.dspark.*`), present when the pack carries it.
struct DsparkGeometry {
    bool present = false;
    int n_layers = 0, block = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0, window = 0, markov_rank = 0;
    float rope_base = 1e6f, rms_eps = 1e-6f;
    std::vector<int> targets;        ///< visits whose input it reads; the last is the visit count (the final normed state)
    std::vector<bool> swa;           ///< per drafter layer
};

struct TensorRef {
    std::string name;
    int type = 0;
    std::vector<uint64_t> shape;     ///< GGUF order, innermost first
    uint64_t offset = 0;             ///< absolute file offset
    uint64_t bytes = 0;              ///< 0: a type this engine does not size
    const uint8_t* mem = nullptr;    ///< the bytes the Pack holds instead of the file's (a GGUF's transposed table)

    /// Sets `it`'s source to the tensor's bytes: its range of plan file `file`, or the bytes the Pack holds.
    void source(int file, core::LoadItem& it) const;
};

/// How the kernels read a dense tensor.
enum class Form { F32, Bf16, Q8_0 };

/// A piece of an expert's blob: `bytes` at file offset `off`, or the bytes at `mem` the Pack holds.
struct BlobSpan {
    uint64_t off = 0, bytes = 0;
    const uint8_t* mem = nullptr;
};

/// A layer's experts.  An expert's blob is its gate rows, its up rows, its down rows and, for NVFP4, a 16-byte tail
/// {s_gate, s_up, s_down, 0} (kernels' native_expert_layout).  A pack stores the layer's blobs one after another from
/// `offset`; a GGUF stores every expert's gate and up rows in one tensor (from `offset`, `gu_bytes` an expert) and their
/// down rows in another (from `d_offset`, `d_bytes` an expert), and the Pack holds the tails it builds from the scales.
struct ExpertLayer {
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0;                   ///< absolute file offset
    uint64_t blob_bytes = 0;
    uint64_t gu_bytes = 0, d_offset = 0, d_bytes = 0;   ///< a GGUF's (gu_bytes 0: a pack)
    const uint8_t* tails = nullptr;        ///< a GGUF's NVFP4 tails, 16 bytes an expert
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    DsparkGeometry dspark;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;      ///< per layer
    core::FileRange vision;                ///< the vision/mmproj section (empty when the pack carries none)
    std::map<std::string, std::vector<uint8_t>> held;   ///< bytes built at open (TensorRef::mem, ExpertLayer::tails)

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int layer, const char* suffix) const { return find("blk." + std::to_string(layer) + "." + suffix); }
    /// Sets `it`'s source and conversion so that it loads tensor `name`, of `values` values, as `form`: F32 as it is;
    /// BF16 as it is, or from F32 whose values are BF16's (llama.cpp's file widens them); Q8_0 as it is, or rounded from
    /// BF16 as it loads (the rounding the pack's converter does).  The destination is the caller's.  False with `err` when
    /// the tensor is missing or is none of those.
    bool source(const std::string& name, Form form, uint64_t values, int file, core::LoadItem& it, std::string& err) const;
    /// Whether layer `l`'s blobs are one range of the file: n_expert blobs from experts[l].offset.
    bool experts_contiguous(int l) const;
    /// Appends the pieces of bytes [at, at + n) of expert `e`'s blob in layer `l` to `out`, in order.
    void expert_spans(int l, int e, uint64_t at, uint64_t n, std::vector<BlobSpan>& out) const;
};

/// The visits of a model with `layers` layers whose layers [start, end) run `passes` times: the prelude, the passes, the
/// coda; the first pass and the layers outside the loop take slots 0 .. layers - 1, each later pass of a core layer one
/// slot after them.
std::vector<Visit> loop_visits(int layers, int start, int end, int passes);

/// Opens the pack or the GGUF and fills `p`. False with `err` when it is not Inkling 2, is malformed or has a layout this
/// engine does not run.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace granite::models::inkling2
