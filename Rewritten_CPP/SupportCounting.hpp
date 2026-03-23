#pragma once

#include "FinalRule.hpp"
#include "RdfIndexes.hpp"

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class SupportCounting {
public:
    // Packed (s, p, o) triple key for used-atoms tracking
    struct UsedTriple {
        int s, p, o;
        bool operator==(const UsedTriple& other) const {
            return s == other.s && p == other.p && o == other.o;
        }
    };

    struct UsedTripleHash {
        std::size_t operator()(const UsedTriple& t) const {
            // Combine hashes of s, p, o
            std::size_t h = std::hash<int>{}(t.s);
            h ^= std::hash<int>{}(t.p) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(t.o) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    // VariableMap equivalent: tracks variable bindings, used constants, and used atoms
    struct BindingMap {
        std::unordered_map<int, int> varToConst;           // variable ID -> entity ID
        std::unordered_set<int> usedConstants;              // constants already bound (for injective check)
        std::unordered_set<UsedTriple, UsedTripleHash> usedAtoms;  // ground triples already used
        bool injectiveMapping = true;

        // Check if a constant is already used by some variable
        bool containsConstant(int constId) const {
            return usedConstants.count(constId) > 0;
        }

        // Check if a ground triple is already used
        bool containsAtom(int s, int p, int o) const {
            return usedAtoms.count({s, p, o}) > 0;
        }

        // Record a ground triple as used, and record any new constants
        void addAtom(int s, int p, int o) {
            usedAtoms.insert({s, p, o});
        }

        // Bind a variable and track the constant
        void bindVar(int varId, int constId) {
            varToConst[varId] = constId;
            usedConstants.insert(constId);
        }

        // Check if a variable is bound
        bool isBound(int varId) const {
            return varToConst.count(varId) > 0;
        }

        // Get binding or -1
        int getBinding(int varId) const {
            auto it = varToConst.find(varId);
            return it != varToConst.end() ? it->second : -1;
        }
    };

    explicit SupportCounting(const RdfIndexes& indexes);

    int countSupport(FinalRule& rule) const;

private:
    const RdfIndexes& indexes_;

    bool matchHeadAtom(
        const Atom& head,
        int subjectId,
        int objectId,
        BindingMap& binding
    ) const;

    bool bodyExists(
        const std::vector<Atom>& body,
        const BindingMap& initialBinding
    ) const;

    bool bodyExistsRecursive(
        const std::vector<Atom>& body,
        std::vector<bool>& used,
        BindingMap& binding,
        int matchedCount
    ) const;

    int scoreAtom(
        const Atom& atom,
        const BindingMap& binding
    ) const;

    int chooseBestAtomIndex(
        const std::vector<Atom>& body,
        const std::vector<bool>& used,
        const BindingMap& binding
    ) const;

    int predicatePairCount(int predicate) const;

    std::vector<BindingMap> extendBindingForAtom(
        const Atom& atom,
        const BindingMap& binding
    ) const;
};