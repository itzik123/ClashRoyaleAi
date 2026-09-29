// The Reflex Lab engine for the browser: an embind layer over LabEngine
// (lab_engine.h), built by web/lab/engine/build.ps1 into engine.js +
// engine.wasm. Every method is a thin forward, so the page, the native CLI
// and the curation suite all run the same code; tools/lab/parity.mjs checks
// the WASM build against the native one, try for try.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "lab_engine.h"

using emscripten::val;

namespace {

std::string cardsJson(const lab::LabEngine& e, val ids) {
    return e.cardsJson(emscripten::vecFromJSArray<int>(ids));
}

// Int32Array of (spawn, cell, delay) triples -> Float32Array of 2n: tower HP
// lost and defender survival, interleaved (lab_cli serve's batch layout).
val rolloutBatch(const lab::LabEngine& e, val triples) {
    const std::vector<int> sca = emscripten::convertJSArrayToNumberVector<int>(triples);
    const int n = static_cast<int>(sca.size() / 3);
    std::vector<float> dmg(n), surv(n), out(2 * size_t(n));
    e.rolloutBatch(sca.data(), n, dmg.data(), surv.data());
    for (int i = 0; i < n; ++i) { out[2 * i] = dmg[i]; out[2 * i + 1] = surv[i]; }
    // Copied into a fresh JS array: a view onto `out` would dangle.
    val result = val::global("Float32Array").new_(out.size());
    result.call<void>("set", val(emscripten::typed_memory_view(out.size(), out.data())));
    return result;
}

}  // namespace

EMSCRIPTEN_BINDINGS(reflex_lab) {
    emscripten::class_<lab::LabEngine>("LabEngine")
        .constructor<unsigned>()
        .function("cardsJson", &cardsJson)
        .function("arenaJson", &lab::LabEngine::arenaJson)
        .function("setMatchup", &lab::LabEngine::setMatchup)
        .function("rolloutBatch", &rolloutBatch)
        .function("framesJson", &lab::LabEngine::framesJson)
        .function("liveStart", &lab::LabEngine::liveStart)
        .function("liveStepJson", &lab::LabEngine::liveStepJson)
        .function("livePlace", &lab::LabEngine::livePlace)
        .function("liveStateJson", &lab::LabEngine::liveStateJson);
}
