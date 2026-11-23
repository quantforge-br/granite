// src/models/inkling2/model.hpp - a loaded Inkling 2 pack: every weight on the main card (the projections of a layer fused
// into one Q8_0 matrix, the experts of a layer in one run of blobs at a stride of whole blocks in a reserved address range,
// the head rounded to Q8_0 as it loads), the token embedding in mapped host memory. With two cards (tensor parallelism)
// each holds every dense weight and its half of every expert (cards(), layer(l, card)).
#pragma once

#include "granite/model/adapter.hpp"
#include "models/inkling2/dspark.hpp"   // DSPARK HOOK
#include "models/inkling2/pack.hpp"
#include "models/inkling2/spill.hpp"
#include "models/inkling2/vision_encoder.hpp"   // [vision]

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

constexpr int64_t kDefaultContext = 8192;

/// The four loop controllers of a core layer: the query, key, value and MLP inputs' messages.
constexpr int kStreams = 4;

struct LayerWeights {
    uint8_t* proj = nullptr;                 ///< Q8_0 [kProj][n_embd]: q, r, gate, sink, k, k taps, v, v taps
    uint8_t* out = nullptr;                  ///< Q8_0 [n_embd][3072]
    /// tensor parallelism: Q8_0 [n_embd][3072 / cards()], the input columns of the card's decode heads (its key/value
    /// groups'), copied from `out` once it has landed (attach)
    uint8_t* out_part = nullptr;
    const float* attn_norm = nullptr;        ///< sliding-window layers
    const float* q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;   ///< full layers
    const float* q_clip = nullptr, *r_clip = nullptr, *rel_table = nullptr;
    const float* k_conv = nullptr, *v_conv = nullptr, *attn_post_conv = nullptr, *ffn_post_conv = nullptr;
    const float* attn_query = nullptr;       ///< [1 or 3][n_embd]; null on layer 0
    const float* ffn_query = nullptr;
    const float* ffn_norm = nullptr;
    const float* router = nullptr;           ///< F32 [n_expert][n_embd]
    const float* router_bias = nullptr;
    uint8_t* experts = nullptr;              ///< n_expert blobs, expert e's at experts + e * expert_stride
    int gu_type = 0, d_type = 0;
    int64_t n_ff = 0;                        ///< the blobs' intermediate width: the experts', or a card's half (cards())
    size_t blob_bytes = 0;                   ///< a blob's bytes at that width (native_expert_layout)
    size_t expert_stride = 0;                ///< blob_bytes rounded up to whole blocks of its type and 16 bytes
    unsigned long long* blob_table = nullptr;   ///< device [n_expert]: each expert's blob address on the card (its copy's), 0: in RAM
    int n_spill = 0;                         ///< experts in RAM (spill.hpp): the layer's last slots
    const int32_t* slot_map = nullptr;       ///< device [n_expert]: each expert's slot in `experts`; null: the expert's own
    const uint8_t* spill_host = nullptr;     ///< the RAM experts' blobs in slot order (pinned), null without a spill
    const float* msg_query = nullptr;        ///< core layers: [kStreams][n_embd]
    const uint16_t* ctrl_down[kStreams] = {};   ///< BF16 [gate_rank][2 n_embd]
    const uint16_t* ctrl_gf[kStreams] = {};     ///< BF16 [2 n_embd][gate_rank]: the gain rows, then the forget rows
};

class InklingModel final : public model::LoadedModel {
public:
    InklingModel();
    ~InklingModel() override;
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    /// Layer l's weights on card `card`.
    const LayerWeights& layer(int l, int card = 0) const { return layers_[card][(size_t) l]; }
    /// The head, Q8_0 [n_vocab][n_embd], on card `card` from row head_first(card): the first card holds every row (the
    /// drafter's fold reads them all); with tensor parallelism the second holds rows [head_first(1), n_vocab), which a
    /// window's head runs there.
    const uint8_t* head(int card = 0) const { return head_[card]; }
    int head_first(int card) const { return card == 0 ? 0 : head_lo_; }
    const uint16_t* embedding() const { return embd_dev_; }    ///< BF16 [n_vocab][n_embd], mapped host memory
    const float* embed_norm(int card = 0) const { return embed_norm_[card]; }
    const float* output_norm(int card = 0) const { return output_norm_[card]; }
    const float* output_query(int card = 0) const { return output_query_[card]; }
    /// [core layers][kStreams][n_embd]: every core layer's message queries, layer loop_start first, on card `card`
    const float* msg_queries(int card = 0) const { return msg_queries_[card]; }
    int device() const { return device_; }
    /// Tensor parallelism (docs/models/inkling2.md): with two cards each holds every layer's attention, glue and router
    /// weights and its half of every expert, card c the gate and up rows [c F, c F + F) and the down matrix's columns
    /// there (F = layer(l, c).n_ff, half the experts' width). Both run every visit on the same rows, and each adds the
    /// two cards' sums of their halves in card order, so both hold the same bits. Each card runs half of the head's rows
    /// (head()); the sampler, the drafter and the image encoder are the first card's. One card: card 0, whole experts.
    int cards() const { return second_ >= 0 ? 2 : 1; }
    int card_device(int card) const { return card == 0 ? device_ : second_; }
    int sessions() const { return flags_.sessions > 0 ? flags_.sessions : 1; }
    const model::LoadRequest& flags() const { return flags_; }
    bool cache_fp8() const { return fp8_; }
    /// [vision] The image encoder: staged with --vision, else it refuses pictures. One per model; ENC runs between requests
    /// and windows, so nothing reads the experts while it holds their memory (lend_room).
    ImageEncoder& encoder() const { return vis_; }
    /// What the sessions share (session.cpp's Shared): made by the first, gone with the last.
    std::weak_ptr<void>& shared() { return shared_; }
    const DsparkWeights* dspark() const { return dspark_.get(); }   // DSPARK HOOK: null without a drafter or with --spec 1
    /// Whether expert e of layer l is on the card (every one, unless the context asked for made some spill to RAM).
    bool on_card(int l, int e) const {
        return spill_.slot.empty() || spill_.slot[(size_t) l].empty() ||
               spill_.slot[(size_t) l][(size_t) e] < geo().n_expert - spill_.n_spill[(size_t) l] ||
               (!copy_.empty() && copy_[(size_t) l * geo().n_expert + e]);
    }
    /// The elastic spill: the experts the asked context sent to RAM come back to the card as copies while its caches back
    /// only the positions in use. On with a spill the context made (not --expert-cache) and a context above the elastic
    /// KV's base (core/elastic_kv.hpp).
    bool elastic() const { return elastic_; }
    /// Copies RAM experts to the card, the most routed first, while `keep_free` bytes stay free; their blob table entries
    /// point at the copies (decode reads them there; a prompt chunk's staging is unchanged). Bytes copied.
    uint64_t promote(uint64_t keep_free);
    /// Sends copies back, the least routed first, until at least `bytes` are freed: their entries go back to 0, the CPU
    /// computes them again (on `stream`, which is waited for before the copies are freed). Bytes freed.
    uint64_t demote(uint64_t bytes, void* stream);
    int64_t promoted() const { return n_copy_; }
    /// The free VRAM the session holds the card at from now on (VRAM / POST /v1/vram; 0: its own spare only): a grow of
    /// the caches and a promote keep at least this much free. `lend(true)` gives every card copy back for a sleep and holds them off (nothing promotes)
    /// until `lend(false)`.
    uint64_t reserve() const { return reserve_; }
    bool lent() const { return lent_; }
    uint64_t held_free(uint64_t base) const { return std::max(base, reserve_); }
    /// Resizes the card's copies so that `free_bytes` are free: copies go back to RAM when short, return while there is
    /// room. False when even no copy at all leaves that much.
    bool resize_reserve(uint64_t free_bytes, void* stream);
    void lend(bool low, void* stream);
    /// A prompt chunk's staging of layer l's RAM experts from their card copies, when every one has a copy (false: none
    /// issued; the caller copies them from RAM).
    bool stage_from_card(int l, void* stream) const;
    const SpillPlan& spill() const { return spill_; }
    /// Expert e of layer l's blob in RAM when the CPU computes it (it spilled and has no card copy; its blob table entry is
    /// 0), else null (cpu_experts.hpp).
    const uint8_t* ram_blob(int l, int e) const;
    /// The layers with an expert the CPU computes, bit l for layer l: what a window's launches depend on.
    uint64_t cpu_layers() const;
    /// [vision] Leaves at least `bytes` free above the card's spare for the image encoder's region: the experts' card copies
    /// go back to RAM first (demote), then whole layers' expert memory goes to RAM, the last layers first, its addresses kept
    /// (the range stays reserved). Every stream is idle on return. False with `err` when even all of it is short (nothing
    /// lent then).
    bool lend_room(uint64_t bytes, std::string& err);
    /// [vision] The lent layers' memory comes back at the same addresses with the same bytes, then the copies (promote).
    void refill();

private:
    float* small(const std::string& name, uint64_t floats, core::LoadPlan& reads, int file, std::string& err,
                 float* dst = nullptr);
    uint16_t* small_bf16(const std::string& name, uint64_t values, core::LoadPlan& reads, int file, std::string& err,
                         uint16_t* dst = nullptr);
    /// Card `card`'s dense weights of every layer and the experts' layout there (its device, its plan's file `file`); their
    /// bytes into `dense` on the first.
    model::Result plan_layers(int card, core::LoadPlan& to, int file, uint64_t& dense);
    /// Every layer's experts, the ones `spill_` sends to RAM there; their bytes on the card into `experts`.
    model::Result load_experts(core::LoadPlan& reads, int file, uint64_t& experts);
    /// Tensor parallelism: each card's half of every layer's experts (the down matrices staged in RAM, then
    /// place_down_halves); their bytes on both cards into `experts`.
    model::Result load_expert_halves(core::LoadPlan& reads, int file, uint64_t& experts);
    /// Each card's half of the down matrices and their tails, from the whole ones the reads staged in RAM (a step of the
    /// plan, once the reads have landed).
    bool place_down_halves(std::string& err);
    /// Two cards: whether they run tensor parallelism (sets second_), else the first runs alone (logged).
    void plan_cards(const model::LoadRequest& req);

    Pack pack_;
    model::ModelDesc desc_;
    std::vector<LayerWeights> layers_[2];   ///< each card's (the second: tensor parallelism)
    std::vector<void*> dev_;                 ///< every device allocation, freed with the model
    uint8_t* head_[2] = {};
    int head_lo_ = 0;                        ///< the first head row the second card holds
    uint16_t* embd_host_ = nullptr;          ///< pinned, mapped
    uint16_t* embd_dev_ = nullptr;           ///< its device address
    const float* embed_norm_[2] = {};
    const float* output_norm_[2] = {};
    const float* output_query_[2] = {};
    float* msg_queries_[2] = {};
    int second_ = -1;                        ///< tensor parallelism's second card (-1: one card)
    // tensor parallelism: every expert's whole down rows and tail as the reads land them, [layer][expert] from each
    // layer's down_stage_at_, freed once both cards have their halves (place_down_halves)
    std::unique_ptr<uint8_t[]> down_stage_;
    std::vector<uint64_t> down_stage_at_;
    model::LoadRequest flags_;
    int device_ = 0, fd_ = -1;
    int ldev_ = 0;                           ///< the card the weights being planned load to
    bool fp8_ = false;
    int64_t ctx_ = kDefaultContext;
    mutable ImageEncoder vis_;   // [vision]
    std::unique_ptr<DsparkWeights> dspark_;   // DSPARK HOOK
    std::weak_ptr<void> shared_;
    // each layer's experts in a reserved range with its own backing (a spilled layer's RAM slots over the staging they share,
    // spill.hpp; without virtual memory management a layer that keeps every expert has a plain allocation), the RAM copies
    struct Range {
        uint8_t* va = nullptr;
        uint64_t bytes = 0, own = 0;
        unsigned long long handle = 0;   ///< the own backing; 0 while lent (lend_room)
    };
    SpillPlan spill_;
    std::vector<Range> ranges_;
    std::vector<unsigned long long> staging_;   ///< granule-sized pieces (the driver maps whole allocations only)
    uint8_t* spill_host_ = nullptr;
    // the elastic spill: the RAM experts the most routed first, each one's card copy (null: none)
    bool elastic_ = false;
    uint64_t reserve_ = 0;
    bool lent_ = false;
    std::vector<std::pair<int32_t, int32_t>> spill_order_;
    std::vector<void*> copy_;
    int64_t n_copy_ = 0;
    std::vector<int> layer_copies_;   ///< [layer] RAM experts with a card copy
    // [vision] the experts' memory lent to the image encoder: each lent range's bytes, in RAM until refill
    struct LentRange {
        size_t range = 0;
        std::unique_ptr<uint8_t[]> bytes;
    };
    std::vector<LentRange> lent_ranges_;
    int64_t lend_copies_ = 0;   ///< card copies demoted for the encoder
    double lend_ms_ = 0;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(InklingModel& model, std::string& err);

/// The device bytes a session takes for `context` positions (fp8: one byte a cached value), its drafter aside.
uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8);
/// The device bytes every session shares once on a card: the forward's scratch, the prompt path, the drafter's tap (the
/// first card's).
uint64_t shared_bytes(const Geometry& g, bool drafter);

}  // namespace basalt::models::inkling2
