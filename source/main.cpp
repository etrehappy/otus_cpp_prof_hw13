#include <iostream>
#include <fstream>
#include <algorithm> 
#include <numeric>   

#include "helpers.h"
#include "logreg_classifier.h"

void FillContainer(std::vector<LogregClassifier>& container, char* argv)
{
    LogregClassifier::coef_t coefs;

    std::ifstream file(argv);

    while (read_coefs(file, coefs))
    {
        container.emplace_back(LogregClassifier(coefs));
    }

    file.close();
}

float GetPredicts(std::istream& data, float& predict_number, const std::vector<LogregClassifier>& classifier_vec)
{
    LogregClassifier::features_t features{};    
    float correct_predict_number{}, max_probability{};
    int correct_type{}, predicted_type{};
 
    features.reserve(1500);

    std::vector<int> indices(classifier_vec.size());
    std::iota(indices.begin(), indices.end(), 0);

    while (read_features(data, features, correct_type))
    {
        ++predict_number;
        max_probability = -1;
        predicted_type = 0;

        indices.clear(); 
        indices.resize(classifier_vec.size()); 
        std::iota(indices.begin(), indices.end(), 0);

        predicted_type = *std::max_element(indices.begin()
            , indices.end()
            , [&classifier_vec, &features](int i, int j) 
            {
                bool compare_probabilities = classifier_vec[i].predict_proba(features) 
                    < classifier_vec[j].predict_proba(features);

                return compare_probabilities;
              }
        );

        if (classifier_vec[predicted_type].predict_proba(features) > max_probability)
        {
            max_probability = classifier_vec[predicted_type].predict_proba(features);
        }

        if (predicted_type == correct_type)
            correct_predict_number++;
    }    

    return correct_predict_number;
}


float GetAccurcy(float correct_predict_number, float predict_number)
{
    float accurcy{0};

    if (predict_number > 0)
    {
        accurcy = correct_predict_number / predict_number;
    }

    return accurcy;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Error args" << std::endl;
        return 1;
    }

    std::ifstream data{argv[1]};
    std::vector<LogregClassifier> classifier_vec{};        
    float predict_number{}, correct_predict_number{};

    FillContainer(classifier_vec, argv[2]); 
    correct_predict_number = GetPredicts(data, predict_number, classifier_vec);
        
    std::cout << GetAccurcy(correct_predict_number, predict_number) << std::endl;

    return 0;
}
