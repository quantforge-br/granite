# The models a build contains. Adding a model is one word in GRANITE_BUILTIN_MODELS plus its own library under
# src/models/<arch> (named model_<arch>, in the namespace granite::models::<arch>, with make_adapter()), gated by
# `if(GRANITE_MODEL_<ARCH>)`. A model left out is not compiled, not linked and not registered.

# Every model that ships: the architecture ids, as `general.architecture` names them.
set(GRANITE_BUILTIN_MODELS inkling2)

# granite builds Inkling 2 Mini only by default. "all" follows GRANITE_BUILTIN_MODELS, so a build folder picks up a model
# added later. The cache keeps the word, not the list: a list written as the default would stay in the folder and leave
# every newer model out.
set(GRANITE_MODELS "inkling2" CACHE STRING
    "Models to build: all, or a semicolon list from: ${GRANITE_BUILTIN_MODELS}")
# The resolved list, as a normal variable over the cache entry for the rest of the configure.
if(GRANITE_MODELS STREQUAL "all")
  set(GRANITE_MODELS ${GRANITE_BUILTIN_MODELS})
endif()

list(REMOVE_DUPLICATES GRANITE_MODELS)
string(REPLACE ";" ", " _granite_valid "${GRANITE_BUILTIN_MODELS}")
if(NOT GRANITE_MODELS)
  message(FATAL_ERROR "GRANITE_MODELS is empty: say all or name at least one of ${_granite_valid}")
endif()
foreach(_m IN LISTS GRANITE_MODELS)
  if(NOT _m IN_LIST GRANITE_BUILTIN_MODELS)
    message(FATAL_ERROR "GRANITE_MODELS names '${_m}', which is not a Basalt model. Valid: all, ${_granite_valid}")
  endif()
endforeach()

foreach(_m IN LISTS GRANITE_BUILTIN_MODELS)
  string(TOUPPER ${_m} _up)
  if(_m IN_LIST GRANITE_MODELS)
    set(GRANITE_MODEL_${_up} ON)
  else()
    set(GRANITE_MODEL_${_up} OFF)
  endif()
endforeach()
message(STATUS "granite: models ${GRANITE_MODELS}")

# The registration list the composition root (src/models/builtin.cpp) expands, one entry per built model.
set(_granite_registrations "")
foreach(_m IN LISTS GRANITE_MODELS)
  string(APPEND _granite_registrations "GRANITE_BUILT_MODEL(${_m})\n")
endforeach()
file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/generated/builtin_models.inc CONTENT "${_granite_registrations}" @ONLY)

# The registry names the models in its error for a pack whose model this build left out.
string(REPLACE ";" "," _granite_known "${GRANITE_BUILTIN_MODELS}")
string(REPLACE ";" "," _granite_built "${GRANITE_MODELS}")
