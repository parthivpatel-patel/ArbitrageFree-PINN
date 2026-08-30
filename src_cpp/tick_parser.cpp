#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>

struct OptionsTick {
    long long timestamp;
    std::string contract;
    double bid;
    double ask;
    int volume;
};

class TickParser {
public:
    std::vector<OptionsTick> parse_csv(const std::string& filepath) {
        std::vector<OptionsTick> tick_data;
        tick_data.reserve(1000000);

        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "CRITICAL ERROR: Could not open tick data file at: " << filepath << std::endl;
            return tick_data;
        }

        std::string line;
        std::getline(file, line); // Skip header

        auto start_time = std::chrono::high_resolution_clock::now();

        while (std::getline(file, line)) {
            OptionsTick tick;
            std::stringstream ss(line);
            std::string token;

            try {
                std::getline(ss, token, ',');
                tick.timestamp = std::stoll(token);

                std::getline(ss, tick.contract, ',');

                std::getline(ss, token, ',');
                tick.bid = std::stod(token);

                std::getline(ss, token, ',');
                tick.ask = std::stod(token);

                std::getline(ss, token, ',');
                tick.volume = std::stoi(token);

                if (tick.bid > 0 && tick.ask > 0) {
                    tick_data.push_back(tick);
                }
            } catch (...) {
                continue;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "[BENCHMARK] Successfully parsed " << tick_data.size() 
                  << " ticks in " << elapsed.count() << " seconds!" << std::endl;

        return tick_data;
    }
};

int main() {
    std::cout << "Initializing High-Frequency C++ Parsing Engine..." << std::endl;
    TickParser parser;
    parser.parse_csv("../data/raw_options_ticks.csv");
    std::cout << "Data stream loaded successfully." << std::endl;
    return 0;
}