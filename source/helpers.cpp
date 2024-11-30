#include "helpers.h"

#include <string>
#include <sstream>
#include <iterator>

bool read_coefs(std::istream& stream, std::vector<float>& container)
{
    std::string line;
    if (!std::getline(stream, line)) { return false; }
    
    std::istringstream linestream{line};
    container.clear();   
    float value{};

    while (linestream >> value)
    {
        container.push_back(value);
    }

    return stream.good();
}

bool read_features(std::istream& stream, BinaryClassifier::features_t& container, int& correct_type)
{
    std::string line;
    if (!std::getline(stream, line)) { return false; }

    std::istringstream linestream{line};
    container.clear();

    std::string str;
    if (std::getline(linestream, str, ','))
    {
        correct_type = std::stoi(str);
    }

    while (std::getline(linestream, str, ','))
    {
        container.push_back(std::stof(str));
    }

    return true;
}

