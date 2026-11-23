#include "run.hpp"
#include "session_model.hpp"
#include "../cli/gpu.hpp"
#include "../cli/startup.hpp"

#include "granite/model/adapter.hpp"
#include "granite/models/builtin.hpp"
#include "granite/program/gpu_select.hpp"
#include "granite/program/options.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace granite::program::generate {

namespace cli = granite::program::cli;

int run(int argc, char** argv) {
    granite::models::register_builtin_models();
    std::unique_ptr<granite::program::CommonOptions> options;
    const granite::model::ModelAdapter* adapter = nullptr;
    if (const auto code = cli::start(argc, argv, options, adapter)) return *code;
    granite::program::CommonOptions& common = *options;
    // ---- the GPUs: chosen before any CUDA call, which then sees only them, the main card as device 0
    std::vector<granite::program::GpuCard> chosen_cards;
    const auto layer_stage = [&](double main_gbps, double peer_gbps) {
        return adapter != nullptr && adapter->second_card_runs_layers(common, main_gbps, peer_gbps);
    };
    if (const auto code = cli::select_gpus(common, chosen_cards, (adapter != nullptr ? adapter->max_expert_cards() : granite::model::max_expert_cards()) + 1,
                                         adapter != nullptr && adapter->second_card_reads_prompts(), layer_stage)) return *code;
    // ---- where every loader reads from: the sections of the .basalt file
    if (const auto code = cli::prepare_model_process(common)) return *code;
    if (const std::optional<int> code = adapter->run(common, chosen_cards)) return *code;
    return run_session_model(common, *adapter);
}

}
