#include "FinalRule.hpp"

Term::Term(TermType t, int v) : type(t), value(v) {}

bool Term::isVariable() const {
    return type == TermType::Variable;
}

bool Term::isConstant() const {
    return type == TermType::Constant;
}

Atom::Atom(int predicateId, const Term& subj, const Term& obj)
    : predicate(predicateId), subject(subj), object(obj) {}

Measures::Measures(int supp, int hSize, int hSupp, double hCov)
    : support(supp), headSize(hSize), headSupport(hSupp), headCoverage(hCov) {}

FinalRule::FinalRule(const Atom& headAtom, const std::vector<Atom>& bodyAtoms)
    : head(headAtom), body(bodyAtoms), measures() {}

FinalRule::FinalRule(const Atom& headAtom, const std::vector<Atom>& bodyAtoms, const Measures& m)
    : head(headAtom), body(bodyAtoms), measures(m) {}

void FinalRule::setMeasures(int supportValue, int headSizeValue, int headSupportValue) {
    measures.support = supportValue;
    measures.headSize = headSizeValue;
    measures.headSupport = headSupportValue;

    if (headSizeValue > 0) {
        measures.headCoverage =
            static_cast<double>(supportValue) / static_cast<double>(headSizeValue);
    } else {
        measures.headCoverage = 0.0;
    }
}

std::size_t FinalRule::bodySize() const {
    return body.size();
}

