#pragma once

#include "gpu_common.cuh"
#include "RdfIndexes.hpp"

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════
 *  Per-predicate sorted int2 index (host + device)
 *  Built from CPU_01's RdfIndexes (PredIndex with CSR).
 *  SPO: sorted (subject, object) pairs
 *  POS: sorted (object, subject) pairs
 * ═══════════════════════════════════════════════════════ */

/* ─── host-side GPU index ─── */
struct PredHost {
    std::vector<int2> spo, pos;   /* sorted (key, val) pairs, deduplicated */
    int spo_maxfan = 0;           /* max values sharing one key in SPO     */
    int pos_maxfan = 0;           /* max values sharing one key in POS     */
};

/* ─── device-side pointers ─── */
struct PredDev {
    int2* spo{};  int spo_n{};
    int2* pos{};  int pos_n{};
};

/* ─── helpers ─── */
static bool cmp2(const int2& a, const int2& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

static bool eq2(const int2& a, const int2& b) {
    return a.x == b.x && a.y == b.y;
}

static int maxFanOut(const std::vector<int2>& v) {
    if (v.empty()) return 0;
    int mx = 0, cnt = 1;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i].x == v[i-1].x) ++cnt;
        else { if (cnt > mx) mx = cnt; cnt = 1; }
    }
    return std::max(mx, cnt);
}

/* ─── build GPU host indexes from RdfIndexes (CPU_01 format) ─── */
static std::unordered_map<int, PredHost> build_gpu_index(const RdfIndexes& indexes) {
    std::unordered_map<int, PredHost> out;
    for (auto& [pred, pi] : indexes.predIndexes) {
        PredHost ph;

        /* SPO: PredIndex.spo is already sorted+deduplicated pair<int,int> */
        ph.spo.reserve(pi.spo.size());
        for (auto& [s, o] : pi.spo)
            ph.spo.push_back(make_int2(s, o));
        ph.spo_maxfan = maxFanOut(ph.spo);

        /* POS: reverse the pairs (object, subject), then sort */
        ph.pos.reserve(pi.spo.size());
        for (auto& [s, o] : pi.spo)
            ph.pos.push_back(make_int2(o, s));
        std::sort(ph.pos.begin(), ph.pos.end(), cmp2);
        ph.pos.erase(std::unique(ph.pos.begin(), ph.pos.end(), eq2), ph.pos.end());
        ph.pos_maxfan = maxFanOut(ph.pos);

        out[pred] = std::move(ph);
    }
    return out;
}

/* ─── upload all predicate arrays to GPU global memory ─── */
static std::unordered_map<int, PredDev> upload(
        const std::unordered_map<int, PredHost>& h)
{
    std::unordered_map<int, PredDev> d;
    for (auto& [p, ph] : h) {
        PredDev pd;
        pd.spo_n = (int)ph.spo.size();
        pd.pos_n = (int)ph.pos.size();
        if (pd.spo_n) {
            CUDA_OK(cudaMalloc(&pd.spo, pd.spo_n * sizeof(int2)));
            CUDA_OK(cudaMemcpy(pd.spo, ph.spo.data(),
                               pd.spo_n * sizeof(int2), cudaMemcpyHostToDevice));
        }
        if (pd.pos_n) {
            CUDA_OK(cudaMalloc(&pd.pos, pd.pos_n * sizeof(int2)));
            CUDA_OK(cudaMemcpy(pd.pos, ph.pos.data(),
                               pd.pos_n * sizeof(int2), cudaMemcpyHostToDevice));
        }
        d[p] = pd;
    }
    return d;
}

/* ─── free all device memory ─── */
static void free_dev(std::unordered_map<int, PredDev>& d) {
    for (auto& [_, pd] : d) {
        if (pd.spo) cudaFree(pd.spo);
        if (pd.pos) cudaFree(pd.pos);
    }
    d.clear();
}
