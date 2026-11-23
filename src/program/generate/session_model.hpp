#pragma once

#include "granite/model/adapter.hpp"
#include "granite/program/options.hpp"

namespace granite::program::generate {

/// A model served through its sessions, to its exit code.
int run_session_model(granite::program::CommonOptions& o, const granite::model::ModelAdapter& adapter);

}  // namespace granite::program::generate
