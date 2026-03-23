#include "SupportCounting.hpp"

SupportCounting::SupportCounting(const RdfIndexes& indexes)
    : indexes_(indexes) {}

int SupportCounting::countSupport(FinalRule& rule) const {
    // Use the new RdfIndexes methods for proper support counting
    // Step 1: Compute head support and size
    rule.measures.headSize = 0;
    rule.measures.headSupport = 0;
    
    const Atom& head = rule.head;
    int predicate = head.predicate;

    // Compute headSize: total number of triples for this predicate
    auto pIt = indexes_.pso.find(predicate);
    if (pIt == indexes_.pso.end()) {
        rule.setMeasures(0, 0, 0);
        return 0;
    }

    for (const auto& [subject, objects] : pIt->second) {
        rule.measures.headSize += static_cast<int>(objects.size());
    }

    // Compute headSupport based on the head atom's binding pattern
    int headSupport = 0;

    bool subjBound = head.subject.isConstant();
    bool objBound = head.object.isConstant();

    if (!subjBound && !objBound) {
        // p(?x, ?y): headSupport = headSize (all triples for this predicate)
        headSupport = rule.measures.headSize;
    } else if (subjBound && !objBound) {
        // p(c, ?y): headSupport = number of objects reachable from c
        int subjId = head.subject.value;
        auto pIt2 = indexes_.pso.find(predicate);
        if (pIt2 != indexes_.pso.end()) {
            auto sIt = pIt2->second.find(subjId);
            if (sIt != pIt2->second.end()) {
                headSupport = static_cast<int>(sIt->second.size());
            }
        }
    } else if (!subjBound && objBound) {
        // p(?x, c): headSupport = number of subjects reaching c
        int objId = head.object.value;
        auto pIt2 = indexes_.pos.find(predicate);
        if (pIt2 != indexes_.pos.end()) {
            auto oIt = pIt2->second.find(objId);
            if (oIt != pIt2->second.end()) {
                headSupport = static_cast<int>(oIt->second.size());
            }
        }
    } else {
        // p(c1, c2): headSupport = 1 if triple exists, 0 otherwise
        headSupport = indexes_.hasTriple(head.subject.value, predicate, head.object.value) ? 1 : 0;
    }

    rule.measures.headSupport = headSupport;

    if (headSupport == 0) {
        rule.setMeasures(0, rule.measures.headSize, 0);
        return 0;
    }

    // Step 2: Enumerate all head bindings and check body existence
    int support = 0;

    for (const auto& [subjectId, objects] : pIt->second) {
        for (int objectId : objects) {
            BindingMap binding;
            binding.injectiveMapping = true;
            if (!matchHeadAtom(head, subjectId, objectId, binding)) {
                continue;
            }

            // Record the head triple as used (Scala: VariableMap + operator records the atom)
            binding.addAtom(subjectId, predicate, objectId);

            if (bodyExists(rule.body, binding)) {
                support++;
            }
        }
    }

    rule.setMeasures(support, rule.measures.headSize, rule.measures.headSupport);
    return support;
}

bool SupportCounting::matchHeadAtom(
    const Atom& head,
    int subjectId,
    int objectId,
    BindingMap& binding
) const {
    // Injective check: if both are variables and they map to the same constant, reject
    if (binding.injectiveMapping &&
        head.subject.isVariable() && head.object.isVariable() &&
        head.subject.value != head.object.value &&
        subjectId == objectId) {
        return false;
    }

    if (head.subject.isConstant()) {
        if (head.subject.value != subjectId) {
            return false;
        }
    } else {
        binding.bindVar(head.subject.value, subjectId);
    }

    if (head.object.isConstant()) {
        if (head.object.value != objectId) {
            return false;
        }
    } else {
        auto it = binding.varToConst.find(head.object.value);
        if (it != binding.varToConst.end()) {
            if (it->second != objectId) {
                return false;
            }
        } else {
            // Injective check: if objectId is already bound to another variable, reject
            if (binding.injectiveMapping && binding.containsConstant(objectId)) {
                return false;
            }
            binding.bindVar(head.object.value, objectId);
        }
    }

    return true;
}

bool SupportCounting::bodyExists(
    const std::vector<Atom>& body,
    const BindingMap& initialBinding
) const {
    if (body.empty()) {
        return true;
    }

    std::vector<bool> used(body.size(), false);
    BindingMap binding = initialBinding;
    return bodyExistsRecursive(body, used, binding, 0);
}

bool SupportCounting::bodyExistsRecursive(
    const std::vector<Atom>& body,
    std::vector<bool>& used,
    BindingMap& binding,
    int matchedCount
) const {
    if (matchedCount == static_cast<int>(body.size())) {
        return true;
    }

    int bestIndex = chooseBestAtomIndex(body, used, binding);
    if (bestIndex == -1) {
        return false;
    }

    const Atom& atom = body[bestIndex];
    auto extensions = extendBindingForAtom(atom, binding);
    if (extensions.empty()) {
        return false;
    }

    used[bestIndex] = true;

    for (const auto& newBinding : extensions) {
        BindingMap saved = binding;
        binding = newBinding;

        if (bodyExistsRecursive(body, used, binding, matchedCount + 1)) {
            return true;
        }

        binding = saved;
    }

    used[bestIndex] = false;
    return false;
}

int SupportCounting::predicatePairCount(int predicate) const {
    auto pit = indexes_.pso.find(predicate);
    if (pit == indexes_.pso.end()) {
        return 0;
    }

    int count = 0;
    for (const auto& [s, objects] : pit->second) {
        (void)s;
        count += static_cast<int>(objects.size());
    }
    return count;
}

int SupportCounting::scoreAtom(const Atom& atom, const BindingMap& binding) const {
    const bool subjectBound =
        atom.subject.isConstant() ||
        (atom.subject.isVariable() && binding.varToConst.count(atom.subject.value));

    const bool objectBound =
        atom.object.isConstant() ||
        (atom.object.isVariable() && binding.varToConst.count(atom.object.value));

    if (subjectBound && objectBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : binding.varToConst.at(atom.subject.value);
        int o = atom.object.isConstant() ? atom.object.value : binding.varToConst.at(atom.object.value);
        return indexes_.hasTriple(s, atom.predicate, o) ? 1 : 0;
    }

    if (subjectBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : binding.varToConst.at(atom.subject.value);
        const auto* objects = indexes_.getObjects(atom.predicate, s);
        return objects ? static_cast<int>(objects->size()) : 0;
    }

    if (objectBound) {
        int o = atom.object.isConstant() ? atom.object.value : binding.varToConst.at(atom.object.value);
        const auto* subjects = indexes_.getSubjects(atom.predicate, o);
        return subjects ? static_cast<int>(subjects->size()) : 0;
    }

    return predicatePairCount(atom.predicate);
}

int SupportCounting::chooseBestAtomIndex(
    const std::vector<Atom>& body,
    const std::vector<bool>& used,
    const BindingMap& binding
) const {
    int bestIndex = -1;
    int bestScore = 0;

    for (std::size_t i = 0; i < body.size(); ++i) {
        if (used[i]) {
            continue;
        }

        int score = scoreAtom(body[i], binding);
        if (bestIndex == -1 || score < bestScore) {
            bestIndex = static_cast<int>(i);
            bestScore = score;
        }
    }

    return bestIndex;
}

std::vector<SupportCounting::BindingMap> SupportCounting::extendBindingForAtom(
    const Atom& atom,
    const BindingMap& binding
) const {
    std::vector<BindingMap> results;

    auto pit = indexes_.pso.find(atom.predicate);
    if (pit == indexes_.pso.end()) {
        return results;
    }

    const bool subjectBound =
        atom.subject.isConstant() ||
        (atom.subject.isVariable() && binding.varToConst.count(atom.subject.value));

    const bool objectBound =
        atom.object.isConstant() ||
        (atom.object.isVariable() && binding.varToConst.count(atom.object.value));

    if (subjectBound && objectBound) {
        int s = atom.subject.isConstant()
            ? atom.subject.value
            : binding.varToConst.at(atom.subject.value);

        int o = atom.object.isConstant()
            ? atom.object.value
            : binding.varToConst.at(atom.object.value);

        if (indexes_.hasTriple(s, atom.predicate, o)) {
            // Scala: containsAtom check
            if (binding.injectiveMapping && binding.containsAtom(s, atom.predicate, o)) {
                return results;
            }
            BindingMap nb = binding;
            nb.addAtom(s, atom.predicate, o);
            results.push_back(std::move(nb));
        }
        return results;
    }

    if (subjectBound) {
        int s = atom.subject.isConstant()
            ? atom.subject.value
            : binding.varToConst.at(atom.subject.value);

        const auto* objects = indexes_.getObjects(atom.predicate, s);
        if (!objects) {
            return results;
        }

        for (int o : *objects) {
            // Resolve the object value for this candidate
            int resolvedO = o;

            if (atom.object.isConstant()) {
                if (atom.object.value != o) {
                    continue;
                }
            } else {
                auto it = binding.varToConst.find(atom.object.value);
                if (it != binding.varToConst.end()) {
                    if (it->second != o) {
                        continue;
                    }
                } else {
                    // New variable binding — Scala: containsConstant check
                    if (binding.injectiveMapping && binding.containsConstant(o)) {
                        continue;
                    }
                }
            }

            // Scala: containsAtom check
            if (binding.injectiveMapping && binding.containsAtom(s, atom.predicate, resolvedO)) {
                continue;
            }

            BindingMap nb = binding;
            if (atom.object.isVariable() && !nb.varToConst.count(atom.object.value)) {
                nb.bindVar(atom.object.value, o);
            }
            nb.addAtom(s, atom.predicate, resolvedO);
            results.push_back(std::move(nb));
        }

        return results;
    }

    if (objectBound) {
        int o = atom.object.isConstant()
            ? atom.object.value
            : binding.varToConst.at(atom.object.value);

        const auto* subjects = indexes_.getSubjects(atom.predicate, o);
        if (!subjects) {
            return results;
        }

        for (int s : *subjects) {
            if (atom.subject.isConstant()) {
                if (atom.subject.value != s) {
                    continue;
                }
            } else {
                auto it = binding.varToConst.find(atom.subject.value);
                if (it != binding.varToConst.end()) {
                    if (it->second != s) {
                        continue;
                    }
                } else {
                    // New variable binding — Scala: containsConstant check
                    if (binding.injectiveMapping && binding.containsConstant(s)) {
                        continue;
                    }
                }
            }

            // Scala: containsAtom check
            if (binding.injectiveMapping && binding.containsAtom(s, atom.predicate, o)) {
                continue;
            }

            BindingMap nb = binding;
            if (atom.subject.isVariable() && !nb.varToConst.count(atom.subject.value)) {
                nb.bindVar(atom.subject.value, s);
            }
            nb.addAtom(s, atom.predicate, o);
            results.push_back(std::move(nb));
        }

        return results;
    }

    // Neither bound: iterate all triples for this predicate
    for (const auto& [s, objects] : pit->second) {
        for (int o : objects) {
            // Scala: subject == object check when both are unbound variables
            if (binding.injectiveMapping &&
                atom.subject.isVariable() && atom.object.isVariable() &&
                atom.subject.value != atom.object.value &&
                s == o) {
                continue;
            }

            BindingMap nb = binding;
            bool valid = true;

            if (atom.subject.isConstant()) {
                if (atom.subject.value != s) {
                    continue;
                }
            } else {
                auto it = nb.varToConst.find(atom.subject.value);
                if (it != nb.varToConst.end()) {
                    if (it->second != s) {
                        continue;
                    }
                } else {
                    // New variable binding — Scala: containsConstant check
                    if (binding.injectiveMapping && binding.containsConstant(s)) {
                        continue;
                    }
                    nb.bindVar(atom.subject.value, s);
                }
            }

            if (atom.object.isConstant()) {
                if (atom.object.value != o) {
                    continue;
                }
            } else {
                auto it = nb.varToConst.find(atom.object.value);
                if (it != nb.varToConst.end()) {
                    if (it->second != o) {
                        continue;
                    }
                } else {
                    // New variable binding — Scala: containsConstant check
                    if (nb.injectiveMapping && nb.containsConstant(o)) {
                        continue;
                    }
                    nb.bindVar(atom.object.value, o);
                }
            }

            // Scala: containsAtom check
            if (nb.injectiveMapping && nb.containsAtom(s, atom.predicate, o)) {
                continue;
            }

            nb.addAtom(s, atom.predicate, o);
            results.push_back(std::move(nb));
        }
    }

    return results;
}
