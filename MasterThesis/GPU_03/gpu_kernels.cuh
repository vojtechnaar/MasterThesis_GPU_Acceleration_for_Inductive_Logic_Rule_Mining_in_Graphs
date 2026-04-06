#pragma once

#include "gpu_common.cuh"
#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════════════
 *  GPU_03: CSR-indexed DFS with __constant__ memory,
 *  template specialization, and warp-cooperative kernels.
 *
 *  Key difference from GPU_01: atom descriptors store CSR
 *  rowPtr/colInd pointers instead of sorted int2 arrays.
 *
 *  Per-atom lookup cost:
 *    GPU_01: lb(arr, n, key) + ub(arr, n, key) = O(log |P|)
 *    GPU_03: rowPtr[key] + rowPtr[key+1]        = O(1)
 *
 *  For rdf:type with |P|≈500K: 38 memory reads → 2 reads.
 *  For existence checks: binary search over entity degree d
 *  instead of predicate size |P|, where d ≪ |P|.
 * ═══════════════════════════════════════════════════════════════ */

/* ─── per-atom descriptor stored in __constant__ memory ─── */
struct AtomGPU {
    const int* rowPtr;    /* CSR row pointers (rows+1 entries)         */
    const int* colInd;    /* CSR column indices (sorted within row)    */
    int        keySlot;   /* variable slot holding the lookup entity   */
    int        valSlot;   /* variable slot to bind (-1 = fully bound)  */
    int        chkSlot;   /* variable slot to check (-1 = enumerating) */
};

__constant__ AtomGPU c_atoms[MAX_BODY];

/* ─── template DFS using CSR O(1) row access + __ldg cache ─── */
template <int L, int N>
__device__ __forceinline__ bool dfs(int v[MAXVAR]) {
    if constexpr (L == N) {
        return true;
    } else {
        const AtomGPU& a = c_atoms[L];
        const int key = v[a.keySlot];

        /* O(1) row access — 2 reads vs 2×O(log n) in GPU_01 */
        const int lo = __ldg(&a.rowPtr[key]);
        const int hi = __ldg(&a.rowPtr[key + 1]);
        if (lo == hi) return false;

        if (a.valSlot < 0) {
            /* fully bound: binary search for target in colInd[lo..hi)
             * Searches over entity degree d, not predicate size |P| */
            const int cv = v[a.chkSlot];
            int l2 = lo, h2 = hi;
            while (l2 < h2) {
                int m = (l2 + h2) >> 1;
                (__ldg(&a.colInd[m]) < cv) ? (l2 = m + 1) : (h2 = m);
            }
            if (l2 == hi || __ldg(&a.colInd[l2]) != cv) return false;
            return dfs<L + 1, N>(v);
        }

        /* enumerate new bindings */
        const int old = v[a.valSlot];
        for (int i = lo; i < hi; ++i) {
            const int nv = __ldg(&a.colInd[i]);
            bool dup = false;
            #pragma unroll
            for (int j = 0; j < MAXVAR; ++j)
                if (j != a.valSlot && v[j] == nv) { dup = true; break; }
            if (dup) continue;
            v[a.valSlot] = nv;
            if (dfs<L + 1, N>(v)) { v[a.valSlot] = old; return true; }
        }
        v[a.valSlot] = old;
        return false;
    }
}

/* ─── standard kernel: 1 thread per head triple ─── */
template <int N>
__global__ void __launch_bounds__(256, 2)
support_kernel(const int2* __restrict__ head, int headN,
               int hS, int hO, int skipSelfLoops,
               unsigned long long* __restrict__ out)
{
    unsigned long long cnt = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < headN; i += blockDim.x * gridDim.x)
    {
        int v[MAXVAR];
        #pragma unroll
        for (int j = 0; j < MAXVAR; ++j) v[j] = -1;
        v[hS] = __ldg(&head[i].x);
        v[hO] = __ldg(&head[i].y);
        if (skipSelfLoops && v[hS] == v[hO]) continue;
        if (dfs<0, N>(v)) ++cnt;
    }
    /* warp-level reduction */
    for (int off = 16; off; off >>= 1)
        cnt += __shfl_down_sync(0xFFFFFFFFu, cnt, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, cnt);
}

/* ─── warp-cooperative kernel: 1 warp per head triple ─── */
/* 32 lanes split the first enumeration atom's range.      */
/* __any_sync early-exit: once any lane finds support, warp moves on. */
template <int N>
__global__ void __launch_bounds__(256, 2)
support_kernel_coop(const int2* __restrict__ head, int headN,
                    int hS, int hO, int skipSelfLoops,
                    unsigned long long* __restrict__ out)
{
    int warpGlobal = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int nWarps     = (gridDim.x * blockDim.x) >> 5;
    int lane       = threadIdx.x & 31;
    int cnt = 0;

    for (int i = warpGlobal; i < headN; i += nWarps) {
        int v[MAXVAR];
        #pragma unroll
        for (int k = 0; k < MAXVAR; ++k) v[k] = -1;
        v[hS] = __ldg(&head[i].x);
        v[hO] = __ldg(&head[i].y);
        if (skipSelfLoops && v[hS] == v[hO]) continue;

        bool found = false;
        const AtomGPU& a0 = c_atoms[0];
        int key0 = v[a0.keySlot];

        /* O(1) row access */
        int lo = __ldg(&a0.rowPtr[key0]);
        int hi = __ldg(&a0.rowPtr[key0 + 1]);

        if (a0.valSlot < 0) {
            /* fully-bound first atom: only lane 0 checks */
            if (lane == 0) {
                int chk = v[a0.chkSlot];
                int l2 = lo, h2 = hi;
                while (l2 < h2) {
                    int m = (l2 + h2) >> 1;
                    (__ldg(&a0.colInd[m]) < chk) ? (l2 = m + 1) : (h2 = m);
                }
                if (l2 < hi && __ldg(&a0.colInd[l2]) == chk) {
                    if constexpr (N == 1) found = true;
                    else found = dfs<1, N>(v);
                }
            }
        } else {
            /* enumerate first atom: 32 lanes split the range */
            int slot = a0.valSlot;
            int span = hi - lo;
            for (int base = 0; base < span && !__any_sync(0xFFFFFFFFu, found);
                 base += 32)
            {
                int idx = base + lane;
                if (idx < span) {
                    int nv = __ldg(&a0.colInd[lo + idx]);
                    bool dup = false;
                    #pragma unroll
                    for (int k = 0; k < MAXVAR; ++k)
                        if (k != slot && v[k] == nv) { dup = true; break; }
                    if (!dup) {
                        v[slot] = nv;
                        if constexpr (N == 1) found = true;
                        else found = dfs<1, N>(v);
                        v[slot] = -1;
                    }
                }
            }
        }
        if (__any_sync(0xFFFFFFFFu, found) && lane == 0) ++cnt;
    }
    if (lane == 0) atomicAdd(out, (unsigned long long)cnt);
}

/* ─── launch dispatch (template N=1..8) ─── */
static void launch_std(int nb, int grd, int blk,
        const int2* h, int hN, int hS, int hO, int ssl, unsigned long long* dr)
{
    switch (nb) {
    case 1: support_kernel<1><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 2: support_kernel<2><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 3: support_kernel<3><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 4: support_kernel<4><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 5: support_kernel<5><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 6: support_kernel<6><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 7: support_kernel<7><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 8: support_kernel<8><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    }
}

static void launch_coop(int nb, int grd, int blk,
        const int2* h, int hN, int hS, int hO, int ssl, unsigned long long* dr)
{
    switch (nb) {
    case 1: support_kernel_coop<1><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 2: support_kernel_coop<2><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 3: support_kernel_coop<3><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 4: support_kernel_coop<4><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 5: support_kernel_coop<5><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 6: support_kernel_coop<6><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 7: support_kernel_coop<7><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 8: support_kernel_coop<8><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    }
}
