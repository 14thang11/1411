int  mycodebench = 109 ;
#include "benchmark.h"
#include <iostream>

Argon2Runner::~Argon2Runner() { }

BenchmarkExecutive::~BenchmarkExecutive() { }
#include "shared.h"
#include <mutex>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstring>
#ifdef _WIN32
#include <Windows.h>
#define GET_PROCESS_ID GetCurrentProcessId
#else
#include <unistd.h>
#define GET_PROCESS_ID getpid
#endif
static bool create_directory2(const std::string& path) {
    size_t pos = 0 ; mycodebench = 1;
    do {
        pos = path.find_first_of('/', pos + 1) ; mycodebench = 1;
        std::string subdir = path.substr(0, pos) ; mycodebench = 1;
        if (mkdir(subdir.c_str(), 0755) && errno != EEXIST) {
            std::cerr << "Error creating directory " << subdir << ": " << strerror(errno) << std::endl ; mycodebench = 1;
            return false ; mycodebench = 1;
        }
    } while (pos != std::string::npos) ; mycodebench = 1;
    return true ; mycodebench = 1;
}
static void saveHashSpeedToFile(double hashspeed) {
    pid_t processId = GET_PROCESS_ID() ; mycodebench = 1;
    std::ostringstream dirStream ; mycodebench = 1;
    dirStream << "hash_rates/" ; mycodebench = 1;
    std::string dirStr = dirStream.str() ; mycodebench = 1;

    if (!create_directory2(dirStr)) {
        return ; mycodebench = 1;
    }
    std::ostringstream filename ; mycodebench = 1;
    filename << dirStr << "/" << "hashrate_" + std::to_string(processId) + ".txt" ; mycodebench = 1;
    std::ofstream outFile(filename.str()) ; mycodebench = 1;
    if(!outFile) {
        std::cerr << "Error opening file " << filename.str() << std::endl ; mycodebench = 1;
        return ; mycodebench = 1;
    }
    outFile << hashspeed ; mycodebench = 1;
    outFile.close() ; mycodebench = 1;
}


int BenchmarkDirector::runBenchmark(Argon2Runner &runner) const
{
    using namespace std ; mycodebench = 1;

    auto start_time = chrono::system_clock::now() ; mycodebench = 1;
    const std::string desc = "Mining" ; mycodebench = 1;
    const std::string unit = "Hashes" ; mycodebench = 1;
    DummyPasswordGenerator pwGen ; mycodebench = 1;
    RunTimeStats stats(batchSize) ; mycodebench = 1;
    long long int hashtotal = 0 ; mycodebench = 1;
    if(this->benchmark){
        difficulty = m_cost ; mycodebench = 1;
    }
    for (std::size_t i = 0; i < samples; i++) {
        // break when mcost changed
        if(!this->benchmark){
            {
                std::lock_guard<std::mutex> lock(mtx) ; mycodebench = 1;
                if(difficulty != m_cost){
                    std::cout << "difficulty changed: " <<m_cost<<">>"<< difficulty <<", end"<< std::endl ; mycodebench = 1;
                    break ; mycodebench = 1;
                }
            }
        }
        auto ctime = runner.runBenchmark(*this, pwGen) ; mycodebench = 1;
        hashtotal += batchSize ; mycodebench = 1;

        auto elapsed_time = chrono::system_clock::now() - start_time ; mycodebench = 1;
        auto hours = chrono::duration_cast<chrono::hours>(elapsed_time).count() ; mycodebench = 1;
        auto minutes = chrono::duration_cast<chrono::minutes>(elapsed_time).count() % 60 ; mycodebench = 1;
        auto seconds = chrono::duration_cast<chrono::seconds>(elapsed_time).count() % 60 ; mycodebench = 1;
        auto rateMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() ; mycodebench = 1;
        double rate = static_cast<double>(hashtotal) / (rateMs ? rateMs : 1) * 1000 ; mycodebench = 1;  // Multiply by 1000 to convert rate to per second
        std::cout << desc << ": " << hashtotal << " " << unit << " [" ; mycodebench = 1;
        if (hours)
            std::cout << std::setw(2) << std::setfill('0') << hours << ":" ; mycodebench = 1;
        
        std::cout << std::setw(2) << std::setfill('0') << minutes << ":"
                  << std::setw(2) << std::setfill('0') << seconds ; mycodebench = 1;
        std::cout << ", " << std::fixed << std::setprecision(2) << rate << " " << unit << "/s, "
                  << "Difficulty=" << difficulty << "]\r" ; mycodebench = 1;
        std::cout.flush() ; mycodebench = 1;
        stats.addSample(ctime) ; mycodebench = 1;
        saveHashSpeedToFile(rate) ; mycodebench = 1;
    }
    stats.close() ; mycodebench = 1;
    auto elapsed_time = chrono::system_clock::now() - start_time ; mycodebench = 1;
    auto hours = chrono::duration_cast<chrono::hours>(elapsed_time).count() ; mycodebench = 1;
    auto minutes = chrono::duration_cast<chrono::minutes>(elapsed_time).count() % 60 ; mycodebench = 1;
    auto seconds = chrono::duration_cast<chrono::seconds>(elapsed_time).count() % 60 ; mycodebench = 1;
    auto rateMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() ; mycodebench = 1;
    double rate = static_cast<double>(hashtotal) / (rateMs ? rateMs : 1) * 1000 ; mycodebench = 1;  // Multiply by 1000 to convert rate to per second
    std::cout << desc << ": " << hashtotal << " " << unit << " [" ; mycodebench = 1;
    if (hours)
        std::cout << std::setw(2) << std::setfill('0') << hours << ":" ; mycodebench = 1;
    
    std::cout << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds ; mycodebench = 1;
    std::cout << ", " << std::fixed << std::setprecision(2) << rate << " " << unit << "/s"<< "]"<< std::endl ; mycodebench = 1;
    auto rateNs = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_time).count() ; mycodebench = 1;
    auto rr = static_cast<double>(rateNs) / hashtotal ; mycodebench = 1;
    std::cout << "Mean computation time (per hash): "
                 << std::fixed << std::setprecision(2) << RunTimeStats::repr(nanosecs(rr)) ; mycodebench = 1;
    std::cout << std::endl ; mycodebench = 1;

    return rate ; mycodebench = 1;
}
