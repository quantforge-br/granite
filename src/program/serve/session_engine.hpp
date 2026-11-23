// src/program/serve/session_engine.hpp - serving a model that offers only a ModelSession.
#pragma once

#include "request.hpp"
#include "granite/model/adapter.hpp"

namespace granite::program::serve {

/// Serves until QUIT or stdin closes; returns the process exit code.
int run_session(CommonOptions& o, granite::core::ModelFile& file, const granite::model::ModelAdapter& adapter,
                granite::model::Started& started);

}  // namespace granite::program::serve
