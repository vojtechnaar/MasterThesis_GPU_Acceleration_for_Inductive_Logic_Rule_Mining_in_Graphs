#pragma once

#include "gpu_common.cuh"
#include "RdfIndexes.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>

/* ═══════════════════════════════════════════════════════════════
 *  GPU_02: Sparse-matrix support counting (TensorLog / RLvLR).
 *
 *  For a chain rule  q1(x,z1) ∧ q2(z1,z2) ∧ ... ∧ qn(zn-1,y) ⇒ p(x,y)
 *  support = nnz(M_q1 × M_q2 × ... × M_qn  .*  M_head)
 *
 *  Each predicate gets two CSR matrices (nEntities × nEntities):
 *    M_spo  — subject→object  (adjacency)
 *    M_pos  — object→subject  (transpose)
 *
 *  Instead of materializing M_body = M_q1 × M_q2 × ... via SpGEMM
 *  (which causes OOM with large intermediates), a fused kernel
 *  iterates over head triples (s,o) and checks whether the body
 *  chain connects s → o, computing support with O(1) extra memory.
 * ═══════════════════════════════════════════════════════════════ */

/* ─── device-side CSR matrix ─── */
struct DevCSR {
    int   rows = 0, cols = 0;
    int64_t nnz = 0;
    int*    d_rowPtr = nullptr;   /* rows+1 */
    int*    d_colInd = nullptr;   /* nnz    */
    bool ownsMemory = false;
};

/* ─── free device CSR ─── */
static void free_dev_csr(DevCSR& d) {
    if (d.ownsMemory) {
        if (d.d_rowPtr) cudaFree(d.d_rowPtr);
        if (d.d_colInd) cudaFree(d.d_colInd);
    }
    d = {};
}

/* ─── build & upload a CSR matrix for one predicate ─── */
static DevCSR build_pred_csr(const std::vector<std::pair<int,int>>& pairs,
                              int nEntities, bool transpose)
{
    /* make sorted (row, col) pairs */
    std::vector<std::pair<int,int>> sorted;
    sorted.reserve(pairs.size());
    for (auto& [s, o] : pairs) {
        if (transpose) sorted.push_back({o, s});
        else           sorted.push_back({s, o});
    }
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    int nnz = (int)sorted.size();

    /* build host CSR arrays */
    std::vector<int> rowPtr(nEntities + 1, 0);
    std::vector<int> colInd(nnz);

    for (int i = 0; i < nnz; ++i) {
        rowPtr[sorted[i].first + 1]++;
        colInd[i] = sorted[i].second;
    }
    for (int i = 1; i <= nEntities; ++i)
        rowPtr[i] += rowPtr[i - 1];

    /* upload to GPU */
    DevCSR d;
    d.rows = nEntities;
    d.cols = nEntities;
    d.nnz  = nnz;
    d.ownsMemory = true;

    CUDA_OK(cudaMalloc(&d.d_rowPtr, (nEntities + 1) * sizeof(int)));
    CUDA_OK(cudaMemcpy(d.d_rowPtr, rowPtr.data(),
                        (nEntities + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (nnz > 0) {
        CUDA_OK(cudaMalloc(&d.d_colInd, nnz * sizeof(int)));
        CUDA_OK(cudaMemcpy(d.d_colInd, colInd.data(),
                            nnz * sizeof(int), cudaMemcpyHostToDevice));
    }

    return d;
}

/* ─── per-predicate pair of CSR matrices ─── */
struct PredCSR {
    DevCSR spo;   /* M: subject→object */
    DevCSR pos;   /* M^T: object→subject */
};

/* ─── build all predicate CSR matrices ─── */
static std::unordered_map<int, PredCSR> build_all_csr(
        const RdfIndexes& indexes, int nEntities)
{
    std::unordered_map<int, PredCSR> out;
    for (auto& [pred, pi] : indexes.predIndexes) {
        PredCSR pc;
        pc.spo = build_pred_csr(pi.spo, nEntities, false);
        pc.pos = build_pred_csr(pi.spo, nEntities, true);
        out[pred] = std::move(pc);
    }
    return out;
}

/* ─── free all predicate CSRs ─── */
static void free_all_csr(std::unordered_map<int, PredCSR>& m) {
    for (auto& [_, pc] : m) {
        free_dev_csr(pc.spo);
        free_dev_csr(pc.pos);
    }
    m.clear();
}

/* ═══════════════════════════════════════════════════════════════
 *  Fused chain-support kernel.
 *
 *  For each (s, o) in M_head, checks whether the body chain
 *  connects s → o through intermediate variables, AND all
 *  extra-check atoms are satisfied.
 *
 *  Equivalent to:  support = nnz(M_q1 × M_q2 × ... × M_qn .* M_head)
 *  with additional per-triple checks for non-chain body atoms.
 *
 *  Head triples are compacted into a flat (s,o) pair array for
 *  maximal GPU thread utilization.
 * ═══════════════════════════════════════════════════════════════ */

struct DevChainCSR {
    const int* rp;   /* rowPtr */
    const int* ci;   /* colInd */
};

/*  Extra-check atom (non-chain body atom).
 *  For each head triple (s, o), resolves row & col then checks whether
 *  the predicate's CSR has a matching entry.
 *
 *  rowSrc / colSrc:
 *    0 = head subject (s)
 *    1 = head object  (o)
 *    2 = constant     (use constVal)
 *    3 = free variable (existence: row must be non-empty)
 */
struct DevExtraCheck {
    const int* rp;
    const int* ci;
    int rowSrc;       /* 0=s, 1=o, 2=const */
    int colSrc;       /* 0=s, 1=o, 2=const, 3=free */
    int rowConst;     /* entity ID if rowSrc==2 */
    int colConst;     /* entity ID if colSrc==2 */
};

/* binary search for 'target' in CSR row 'row' */
__device__ __forceinline__
bool bsearch_row(const int* rp, const int* ci, int row, int target)
{
    int lo = __ldg(&rp[row]);
    int hi = __ldg(&rp[row + 1]);
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int v = __ldg(&ci[mid]);
        if (v == target) return true;
        if (v < target) lo = mid + 1;
        else            hi = mid;
    }
    return false;
}

/* check if CSR row 'row' has any entries */
__device__ __forceinline__
bool row_nonempty(const int* rp, int row)
{
    return __ldg(&rp[row + 1]) > __ldg(&rp[row]);
}

/* recursive chain check: does chain[depth..chainLen) connect current → target? */
__device__
bool chain_check(int current, int target,
                 int depth, int chainLen,
                 const DevChainCSR* __restrict__ chain)
{
    const int* rp = chain[depth].rp;
    const int* ci = chain[depth].ci;

    if (depth == chainLen - 1)
        return bsearch_row(rp, ci, current, target);

    int lo = __ldg(&rp[current]);
    int hi = __ldg(&rp[current + 1]);
    for (int i = lo; i < hi; ++i) {
        int z = __ldg(&ci[i]);
        if (chain_check(z, target, depth + 1, chainLen, chain))
            return true;
    }
    return false;
}

/* evaluate one extra-check atom for head triple (s, o) */
__device__ __forceinline__
bool eval_extra(const DevExtraCheck& ec, int s, int o)
{
    int row;
    switch (ec.rowSrc) {
        case 0:  row = s; break;
        case 1:  row = o; break;
        default: row = ec.rowConst; break;
    }
    switch (ec.colSrc) {
        case 3:  return row_nonempty(ec.rp, row);        /* free → existence  */
        case 0:  return bsearch_row(ec.rp, ec.ci, row, s); break;
        case 1:  return bsearch_row(ec.rp, ec.ci, row, o); break;
        default: return bsearch_row(ec.rp, ec.ci, row, ec.colConst); break;
    }
}

/* main fused kernel: 1 thread per head triple (s, o) */
__global__ void fused_support_kernel(
        const int2* __restrict__ headPairs,   /* compact (s,o) array */
        int nPairs,
        int chainLen,
        const DevChainCSR* __restrict__ chain,
        int numFilters,
        const DevChainCSR* __restrict__ filters,
        int numExtras,
        const DevExtraCheck* __restrict__ extras,
        int skipSelfLoops,
        unsigned long long* __restrict__ out)
{
    unsigned long long local = 0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < nPairs; idx += blockDim.x * gridDim.x)
    {
        int2 p = __ldg(&headPairs[idx]);
        int s = p.x, o = p.y;

        /* injective mapping: skip head self-loops */
        if (skipSelfLoops && s == o) continue;

        /* check body chain: does chain connect s → o ? */
        if (chainLen > 0 && !chain_check(s, o, 0, chainLen, chain))
            continue;

        /* check filter atoms: each filter must have (s, o) */
        bool ok = true;
        for (int f = 0; f < numFilters && ok; ++f)
            ok = bsearch_row(filters[f].rp, filters[f].ci, s, o);

        /* check extra atoms: existence / constant / pair lookups */
        for (int e = 0; e < numExtras && ok; ++e)
            ok = eval_extra(extras[e], s, o);

        if (ok) ++local;
    }

    /* warp reduction */
    for (int off = 16; off; off >>= 1)
        local += __shfl_down_sync(0xFFFFFFFFu, local, off);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(out, local);
}

/* ─── compact head triples ─── */
struct CompactHead {
    int2* d_pairs = nullptr;
    int   nPairs  = 0;
};

static CompactHead build_compact_head(const DevCSR& head)
{
    CompactHead ch;
    if (head.nnz == 0) return ch;

    /* download rowPtr and colInd */
    int n = head.rows;
    std::vector<int> rp(n + 1), ci(head.nnz);
    CUDA_OK(cudaMemcpy(rp.data(), head.d_rowPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ci.data(), head.d_colInd, head.nnz * sizeof(int), cudaMemcpyDeviceToHost));

    /* build (s, o) pairs */
    std::vector<int2> pairs;
    pairs.reserve(head.nnz);
    for (int i = 0; i < n; ++i)
        for (int j = rp[i]; j < rp[i + 1]; ++j)
            pairs.push_back(make_int2(i, ci[j]));

    ch.nPairs = (int)pairs.size();
    CUDA_OK(cudaMalloc(&ch.d_pairs, ch.nPairs * sizeof(int2)));
    CUDA_OK(cudaMemcpy(ch.d_pairs, pairs.data(),
                        ch.nPairs * sizeof(int2), cudaMemcpyHostToDevice));
    return ch;
}

static void free_compact_head(CompactHead& ch) {
    if (ch.d_pairs) { cudaFree(ch.d_pairs); ch.d_pairs = nullptr; }
    ch.nPairs = 0;
}

/* ─── host wrapper ─── */
static unsigned long long fused_chain_support(
        const CompactHead& head,
        const std::vector<const DevCSR*>& chainMats,
        const std::vector<const DevCSR*>& filterMats,
        const std::vector<DevExtraCheck>& extraChecks,
        int skipSelfLoops)
{
    if (head.nPairs == 0) return 0;

    int chainLen   = (int)chainMats.size();
    int numFilters = (int)filterMats.size();
    int numExtras  = (int)extraChecks.size();

    /* build host pointer arrays */
    std::vector<DevChainCSR> hChain(chainLen);
    for (int i = 0; i < chainLen; ++i)
        hChain[i] = { chainMats[i]->d_rowPtr, chainMats[i]->d_colInd };

    std::vector<DevChainCSR> hFilter(numFilters);
    for (int i = 0; i < numFilters; ++i)
        hFilter[i] = { filterMats[i]->d_rowPtr, filterMats[i]->d_colInd };

    /* upload to device */
    DevChainCSR* d_chain = nullptr;
    DevChainCSR* d_filter = nullptr;
    DevExtraCheck* d_extras = nullptr;

    if (chainLen > 0) {
        CUDA_OK(cudaMalloc(&d_chain, chainLen * sizeof(DevChainCSR)));
        CUDA_OK(cudaMemcpy(d_chain, hChain.data(),
                            chainLen * sizeof(DevChainCSR), cudaMemcpyHostToDevice));
    }
    if (numFilters > 0) {
        CUDA_OK(cudaMalloc(&d_filter, numFilters * sizeof(DevChainCSR)));
        CUDA_OK(cudaMemcpy(d_filter, hFilter.data(),
                            numFilters * sizeof(DevChainCSR), cudaMemcpyHostToDevice));
    }
    if (numExtras > 0) {
        CUDA_OK(cudaMalloc(&d_extras, numExtras * sizeof(DevExtraCheck)));
        CUDA_OK(cudaMemcpy(d_extras, extraChecks.data(),
                            numExtras * sizeof(DevExtraCheck), cudaMemcpyHostToDevice));
    }

    unsigned long long* d_cnt;
    CUDA_OK(cudaMalloc(&d_cnt, sizeof(unsigned long long)));
    CUDA_OK(cudaMemset(d_cnt, 0, sizeof(unsigned long long)));

    int blk = 256;
    int grd = std::min((head.nPairs + blk - 1) / blk, 2048);

    fused_support_kernel<<<grd, blk>>>(
        head.d_pairs, head.nPairs,
        chainLen, d_chain,
        numFilters, d_filter,
        numExtras, d_extras,
        skipSelfLoops,
        d_cnt);
    CUDA_OK(cudaDeviceSynchronize());

    unsigned long long cnt;
    CUDA_OK(cudaMemcpy(&cnt, d_cnt, sizeof(cnt), cudaMemcpyDeviceToHost));

    cudaFree(d_cnt);
    if (d_chain)  cudaFree(d_chain);
    if (d_filter) cudaFree(d_filter);
    if (d_extras) cudaFree(d_extras);

    return cnt;
}
