#pragma once

#include <vector>


class BinaryClassifier {
public:

    using features_t = std::vector<float>;

    virtual ~BinaryClassifier() {}

    virtual float predict_proba(const features_t&) const = 0;
};