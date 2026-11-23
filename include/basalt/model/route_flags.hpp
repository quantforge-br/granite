// include/basalt/model/route_flags.hpp - the opt-in routing settings a load request carries, which trade accuracy for
// fewer experts that no GPU holds (basalt/kernels/route_lossy.hpp):
//
//   --route-tail-skip [R]   in a decode window an expert no GPU holds is neither copied nor computed when every entry of the
//                           window that names it ranks R or lower by router weight (0-based of the model's k); its entries
//                           get weight 0 and the others keep theirs.  R in 1..k-1; the flag alone = k-1.
//   --route-prior [LAMBDA]  a decode window picks its k experts as if every expert a GPU holds had LAMBDA added to the
//                           score its router takes the top k of (the log probability for a softmax router); the weights stay
//                           the router's own, normalised over the picks as the model normalises them.  LAMBDA in (0, 16];
//                           the flag alone = the model's measured default (its scores' scale differs).
//
// Both are off by default and change the output.  No model in this build takes the flags, so both stay off.
#pragma once

namespace basalt::model {

struct RouteFlags {
    int tail_skip = 0;     ///< R; 0 = off
    float prior = 0.0f;    ///< LAMBDA; 0 = off
};

}  // namespace basalt::model
