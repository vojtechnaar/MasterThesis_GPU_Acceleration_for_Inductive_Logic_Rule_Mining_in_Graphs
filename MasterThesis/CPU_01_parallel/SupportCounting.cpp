#include "SupportCounting.hpp"

SupportCounting::SupportCounting(const RdfIndexes& indexes)
    : indexes_(indexes) {}

// ---- Head matching ----

bool SupportCounting::matchHeadAtom(const Atom& head, int s, int o, Env& env) const {
    // Reject two different variables mapping to the same constant (injective)
    if (head.subject.isVariable() && head.object.isVariable() &&
        head.subject.value != head.object.value && s == o) {
        return false;
    }

    if (head.subject.isConstant()) {
        if (head.subject.value != s) return false;
    } else {
        env.bind(head.subject.value, s);
    }

    if (head.object.isConstant()) {
        if (head.object.value != o) return false;
    } else {
        if (env.isBound(head.object.value)) {
            if (env.resolve(head.object.value) != o) return false;
        } else {
            if (env.containsConstant(o)) return false;
            env.bind(head.object.value, o);
        }
    }

    return true;
}

// ---- Atom scoring (greedy best-atom selection) ----

int SupportCounting::scoreAtom(const Atom& atom, const Env& env) const {
    bool sBound = atom.subject.isConstant() ||
                  (atom.subject.isVariable() && env.isBound(atom.subject.value));
    bool oBound = atom.object.isConstant() ||
                  (atom.object.isVariable() && env.isBound(atom.object.value));

    const PredIndex* pi = indexes_.getPred(atom.predicate);
    if (!pi) return 0;

    if (sBound && oBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : env.resolve(atom.subject.value);
        int o = atom.object.isConstant() ? atom.object.value : env.resolve(atom.object.value);
        return pi->hasTriple(s, o) ? 1 : 0;
    }
    if (sBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : env.resolve(atom.subject.value);
        int cnt = 0;
        pi->spoRange(s, cnt);
        return cnt;
    }
    if (oBound) {
        int o = atom.object.isConstant() ? atom.object.value : env.resolve(atom.object.value);
        int cnt = 0;
        pi->posRange(o, cnt);
        return cnt;
    }
    return pi->totalPairs();
}

int SupportCounting::chooseBestAtom(const Atom* body, int n, uint8_t remaining, const Env& env) const {
    int best = -1;
    int bestScore = 0;
    for (int i = 0; i < n; ++i) {
        if (!(remaining & (1u << i))) continue;
        int sc = scoreAtom(body[i], env);
        if (best < 0 || sc < bestScore) {
            best = i;
            bestScore = sc;
        }
    }
    return best;
}

// ---- Templated DFS with backtracking (zero heap allocation) ----

template<int N>
bool SupportCounting::dfsExists(const Atom* body, uint8_t remaining, Env& env, int matched) const {
    if (matched == N) return true;

    int bestIdx = chooseBestAtom(body, N, remaining, env);
    if (bestIdx < 0) return false;

    const Atom& atom = body[bestIdx];
    uint8_t nextRemaining = remaining & ~(uint8_t(1) << bestIdx);

    const PredIndex* pi = indexes_.getPred(atom.predicate);
    if (!pi) return false;

    bool sBound = atom.subject.isConstant() ||
                  (atom.subject.isVariable() && env.isBound(atom.subject.value));
    bool oBound = atom.object.isConstant() ||
                  (atom.object.isVariable() && env.isBound(atom.object.value));

    // ---- Case 1: both bound ----
    if (sBound && oBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : env.resolve(atom.subject.value);
        int o = atom.object.isConstant() ? atom.object.value : env.resolve(atom.object.value);
        if (!pi->hasTriple(s, o)) return false;
        if (env.containsAtom(s, atom.predicate, o)) return false;

        env.pushAtom(s, atom.predicate, o);
        bool found = dfsExists<N>(body, nextRemaining, env, matched + 1);
        env.popAtom();
        return found;
    }

    // ---- Case 2: subject bound, object unbound ----
    if (sBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : env.resolve(atom.subject.value);
        int cnt = 0;
        const int* objs = pi->spoRange(s, cnt);
        if (!objs) return false;

        for (int i = 0; i < cnt; ++i) {
            int o = objs[i];
            int objVarId = -1;
            int savedObj = 0;

            if (atom.object.isVariable()) {
                objVarId = atom.object.value;
                if (env.isBound(objVarId)) {
                    if (env.resolve(objVarId) != o) continue;
                    objVarId = -1;  // already bound correctly
                } else {
                    if (env.containsConstant(o)) continue;
                    savedObj = env.v[objVarId];
                    env.bind(objVarId, o);
                }
            }

            if (env.containsAtom(s, atom.predicate, o)) {
                if (objVarId >= 0) env.v[objVarId] = savedObj;
                continue;
            }

            env.pushAtom(s, atom.predicate, o);
            bool found = dfsExists<N>(body, nextRemaining, env, matched + 1);
            env.popAtom();
            if (objVarId >= 0) env.v[objVarId] = savedObj;
            if (found) return true;
        }
        return false;
    }

    // ---- Case 3: object bound, subject unbound ----
    if (oBound) {
        int o = atom.object.isConstant() ? atom.object.value : env.resolve(atom.object.value);
        int cnt = 0;
        const int* subjs = pi->posRange(o, cnt);
        if (!subjs) return false;

        for (int i = 0; i < cnt; ++i) {
            int s = subjs[i];
            int subjVarId = -1;
            int savedSubj = 0;

            if (atom.subject.isVariable()) {
                subjVarId = atom.subject.value;
                if (env.isBound(subjVarId)) {
                    if (env.resolve(subjVarId) != s) continue;
                    subjVarId = -1;
                } else {
                    if (env.containsConstant(s)) continue;
                    savedSubj = env.v[subjVarId];
                    env.bind(subjVarId, s);
                }
            }

            if (env.containsAtom(s, atom.predicate, o)) {
                if (subjVarId >= 0) env.v[subjVarId] = savedSubj;
                continue;
            }

            env.pushAtom(s, atom.predicate, o);
            bool found = dfsExists<N>(body, nextRemaining, env, matched + 1);
            env.popAtom();
            if (subjVarId >= 0) env.v[subjVarId] = savedSubj;
            if (found) return true;
        }
        return false;
    }

    // ---- Case 4: neither bound ----
    for (const auto& [s, o] : pi->spo) {
        // Reject s == o for different unbound variables (injective)
        if (atom.subject.isVariable() && atom.object.isVariable() &&
            atom.subject.value != atom.object.value && s == o) {
            continue;
        }

        int subjVarId = -1, objVarId = -1;
        int savedSubj = 0, savedObj = 0;

        // Bind subject
        if (atom.subject.isVariable()) {
            subjVarId = atom.subject.value;
            if (env.isBound(subjVarId)) {
                if (env.resolve(subjVarId) != s) continue;
                subjVarId = -1;
            } else {
                if (env.containsConstant(s)) continue;
                savedSubj = env.v[subjVarId];
                env.bind(subjVarId, s);
            }
        }

        // Bind object
        if (atom.object.isVariable()) {
            objVarId = atom.object.value;
            if (env.isBound(objVarId)) {
                if (env.resolve(objVarId) != o) {
                    if (subjVarId >= 0) env.v[subjVarId] = savedSubj;
                    continue;
                }
                objVarId = -1;
            } else {
                if (env.containsConstant(o)) {
                    if (subjVarId >= 0) env.v[subjVarId] = savedSubj;
                    continue;
                }
                savedObj = env.v[objVarId];
                env.bind(objVarId, o);
            }
        }

        if (env.containsAtom(s, atom.predicate, o)) {
            if (objVarId >= 0) env.v[objVarId] = savedObj;
            if (subjVarId >= 0) env.v[subjVarId] = savedSubj;
            continue;
        }

        env.pushAtom(s, atom.predicate, o);
        bool found = dfsExists<N>(body, nextRemaining, env, matched + 1);
        env.popAtom();
        if (objVarId >= 0) env.v[objVarId] = savedObj;
        if (subjVarId >= 0) env.v[subjVarId] = savedSubj;
        if (found) return true;
    }
    return false;
}

// ---- Dispatch to template ----

bool SupportCounting::bodyExists(const Atom* body, int n, Env& env) const {
    if (n == 0) return true;
    uint8_t all = (uint8_t(1) << n) - 1;
    switch (n) {
        case 1: return dfsExists<1>(body, all, env, 0);
        case 2: return dfsExists<2>(body, all, env, 0);
        case 3: return dfsExists<3>(body, all, env, 0);
        case 4: return dfsExists<4>(body, all, env, 0);
        case 5: return dfsExists<5>(body, all, env, 0);
        case 6: return dfsExists<6>(body, all, env, 0);
        case 7: return dfsExists<7>(body, all, env, 0);
        case 8: return dfsExists<8>(body, all, env, 0);
        default: return false;
    }
}

// ---- Support counting entry point ----

int SupportCounting::countSupport(FinalRule& rule) const {
    const Atom& head = rule.head;
    const PredIndex* pi = indexes_.getPred(head.predicate);
    if (!pi) {
        rule.setMeasures(0, 0, 0);
        return 0;
    }

    rule.measures.headSize = pi->totalPairs();

    bool sBound = head.subject.isConstant();
    bool oBound = head.object.isConstant();

    // Compute headSupport based on head binding pattern
    int headSupport = 0;
    if (!sBound && !oBound) {
        headSupport = pi->totalPairs();
    } else if (sBound && !oBound) {
        int cnt = 0;
        pi->spoRange(head.subject.value, cnt);
        headSupport = cnt;
    } else if (!sBound && oBound) {
        int cnt = 0;
        pi->posRange(head.object.value, cnt);
        headSupport = cnt;
    } else {
        headSupport = pi->hasTriple(head.subject.value, head.object.value) ? 1 : 0;
    }

    rule.measures.headSupport = headSupport;
    if (headSupport == 0) {
        rule.setMeasures(0, rule.measures.headSize, 0);
        return 0;
    }

    // Enumerate head triples efficiently based on binding pattern
    int support = 0;
    const Atom* bodyData = rule.body.data();
    int bodySize = static_cast<int>(rule.body.size());

    auto processHead = [&](int s, int o) {
        Env env;
        env.init();
        if (!matchHeadAtom(head, s, o, env)) return;
        env.pushAtom(s, head.predicate, o);
        if (bodyExists(bodyData, bodySize, env)) {
            ++support;
        }
    };

    if (sBound && oBound) {
        if (pi->hasTriple(head.subject.value, head.object.value)) {
            processHead(head.subject.value, head.object.value);
        }
    } else if (sBound) {
        int cnt = 0;
        const int* objs = pi->spoRange(head.subject.value, cnt);
        if (objs) {
            for (int i = 0; i < cnt; ++i) {
                processHead(head.subject.value, objs[i]);
            }
        }
    } else if (oBound) {
        int cnt = 0;
        const int* subjs = pi->posRange(head.object.value, cnt);
        if (subjs) {
            for (int i = 0; i < cnt; ++i) {
                processHead(subjs[i], head.object.value);
            }
        }
    } else {
        for (const auto& [s, o] : pi->spo) {
            processHead(s, o);
        }
    }

    rule.setMeasures(support, rule.measures.headSize, rule.measures.headSupport);
    return support;
}
