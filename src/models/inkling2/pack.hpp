// src/models/inkling2/pack.hpp - the Inkling 2 .granite pack as the loader sees it: geometry and loop from the meta's
// gguf_kv, the tensor table, the experts layout, and where the vision tower and the drafter sit
// (tools/convert/checkpoint_inkling2.py writes it).
#pragma once

#include "granite/core/model_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace granite::models::inkling2 {

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
    uint64_t bytes = 0;
};

/// A piece of an expert's blob: `bytes` at file offset `off`, or the bytes at `mem` the Pack holds.
struct BlobSpan {
    uint64_t off = 0, bytes = 0;
    const uint8_t* mem = nullptr;
};

/// A layer's experts.  An expert's blob is its gate rows, its up rows, its down rows and, for NVFP4, a 16-byte tail
/// (kernels' native_expert_layout); the pack stores the layer's blobs one after another from `offset`.
struct ExpertLayer {
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0;                   ///< absolute file offset of the first blob
    uint64_t blob_bytes = 0;
};

struct Pack {
    core::ModelFile file;
    Geometry geo;
    DsparkGeometry dspark;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertLayer> experts;      ///< per layer
    core::FileRange vision;                ///< the vision/mmproj section (empty when the pack carries none)

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorRef* find(int layer, const char* suffix) const { return find("blk." + std::to_string(layer) + "." + suffix); }
    /// Whether layer `l`'s blobs are one range of the file: n_expert blobs from experts[l].offset.
    bool experts_contiguous(int l) const;
    /// Appends the pieces of bytes [at, at + n) of expert `e`'s blob in layer `l` to `out`, in order.
    void expert_spans(int l, int e, uint64_t at, uint64_t n, std::vector<BlobSpan>& out) const;
};

/// The visits of a model with `layers` layers whose layers [start, end) run `passes` times: the prelude, the passes, the
/// coda; the first pass and the layers outside the loop take slots 0 .. layers - 1, each later pass of a core layer one
/// slot after them.
std::vector<Visit> loop_visits(int layers, int start, int end, int passes);

/// Opens the pack and fills `p`. False with `err` when it is not an Inkling 2 pack, is malformed or has a layout this
/// engine does not run.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace granite::models::inkling2
