#pragma once

#include <istream>
#include <vector>

#include "classifier.h"

bool read_features(std::istream& stream, BinaryClassifier::features_t& features, int& targetClass);
bool read_coefs(std::istream& stream, std::vector<float>& coefs);