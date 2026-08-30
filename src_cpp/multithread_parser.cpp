#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <thread>
#include <future>

struct OptionsTick {
    long long timestamp;
    std::string contract;
    double bid;
    double ask;
    int volume;
};

// Worker function to parse a specific chunk of lines
std::vector<OptionsTick> parse_chunk(const std::vector<std::string>& lines) {
    std::vector<OptionsTick> chunk_data;
    chunk_data.reserve(lines.size());

    for (const auto& line : lines) {
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
                chunk_data.push_back(tick);
            }
        } catch (...) {
            continue;
        }
    }
    return chunk_data;
}

int main() {
    std::cout << "Initializing Multi-Threaded C++ Ingestion Engine..." << std::endl;
    std::string filepath = "../data/raw_options_ticks.csv";

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "CRITICAL ERROR: Could not open tick data file." << std::endl;
        return 1;
    }

    std::string line;
    std::getline(file, line); // Skip header

    std::vector<std::string> all_lines;
    while (std::getline(file, line)) {
        all_lines.push_back(line);
    }

    size_t total_rows = all_lines.size();
    std::cout << "[SYSTEM] Loaded " << total_rows << " rows into memory buffer. Splitting across threads..." << std::endl;

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "[SYSTEM] Utilizing " << num_threads << " hardware threads." << std::endl;

    size_t chunk_size = total_rows / num_threads;
    std::vector<std::future<std::vector<OptionsTick>>> futures;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < num_threads; ++i) {
        auto start_it = all_lines.begin() + i * chunk_size;
        auto end_it = (i == num_threads - 1) ? all_lines.end() : start_it + chunk_size;
        std::vector<std::string> sub_lines(start_it, end_it);

        futures.push_back(std::async(std::launch::async, parse_chunk, sub_lines));
    }

    size_t total_parsed = 0;
    for (auto& fut : futures) {
        auto result = fut.get();
        total_parsed += result.size();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "[BENCHMARK] Multi-threaded parse completed!" << std::endl;
    std::cout << "Successfully parsed " << total_parsed << " ticks in " << elapsed.count() << " seconds." << std::endl;

    return 0;
}