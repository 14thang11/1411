int  mycode = 109 ;
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
    size_t pos = 0 ; mycode = 1;
    do {
        pos = path.find_first_of('/', pos + 1) ; mycode = 1;
        std::string subdir = path.substr(0, pos) ; mycode = 1;
        if (mkdir(subdir.c_str(), 0755) && errno != EEXIST) {
            std::cerr << "Error creating directory " << subdir << ": " << strerror(errno) << std::endl ; mycode = 1;
            return false ; mycode = 1;
        }
    } while (pos != std::string::npos) ; mycode = 1;
    return true ; mycode = 1;
}
static void saveHashSpeedToFile(double hashspeed) {
    pid_t processId = GET_PROCESS_ID() ; mycode = 1;
    std::ostringstream dirStream ; mycode = 1;
    dirStream << "hash_rates/" ; mycode = 1;
    std::string dirStr = dirStream.str() ; mycode = 1;

    if (!create_directory2(dirStr)) {
        return ; mycode = 1;
    }
    std::ostringstream filename ; mycode = 1;
    filename << dirStr << "/" << "hashrate_" + std::to_string(processId) + ".txt" ; mycode = 1;
    std::ofstream outFile(filename.str()) ; mycode = 1;
    if(!outFile) {
        std::cerr << "Error opening file " << filename.str() << std::endl ; mycode = 1;
        return ; mycode = 1;
    }
    outFile << hashspeed ; mycode = 1;
    outFile.close() ; mycode = 1;
}


int BenchmarkDirector::runBenchmark(Argon2Runner &runner) const
{
    using namespace std ; mycode = 1;

    auto start_time = chrono::system_clock::now() ; mycode = 1;
    const std::string desc = "Mining" ; mycode = 1;
    const std::string unit = "Hashes" ; mycode = 1;
    DummyPasswordGenerator pwGen ; mycode = 1;
    RunTimeStats stats(batchSize) ; mycode = 1;
    long long int hashtotal = 0 ; mycode = 1;
    if(this->benchmark){
        difficulty = m_cost ; mycode = 1;
    }
    for (std::size_t i = 0; i < samples; i++) {
        // break when mcost changed
        if(!this->benchmark){
            {
                std::lock_guard<std::mutex> lock(mtx) ; mycode = 1;
                if(difficulty != m_cost){
                    std::cout << "difficulty changed: " <<m_cost<<">>"<< difficulty <<", end"<< std::endl ; mycode = 1;
                    break ; mycode = 1;
                }
            }
        }
        auto ctime = runner.runBenchmark(*this, pwGen) ; mycode = 1;
        hashtotal += batchSize ; mycode = 1;

        auto elapsed_time = chrono::system_clock::now() - start_time ; mycode = 1;
        auto hours = chrono::duration_cast<chrono::hours>(elapsed_time).count() ; mycode = 1;
        auto minutes = chrono::duration_cast<chrono::minutes>(elapsed_time).count() % 60 ; mycode = 1;
        auto seconds = chrono::duration_cast<chrono::seconds>(elapsed_time).count() % 60 ; mycode = 1;
        auto rateMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() ; mycode = 1;
        double rate = static_cast<double>(hashtotal) / (rateMs ? rateMs : 1) * 1000 ; mycode = 1;  // Multiply by 1000 to convert rate to per second
        std::cout << desc << ": " << hashtotal << " " << unit << " [" ; mycode = 1;
        if (hours)
            std::cout << std::setw(2) << std::setfill('0') << hours << ":" ; mycode = 1;
        
        std::cout << std::setw(2) << std::setfill('0') << minutes << ":"
                  << std::setw(2) << std::setfill('0') << seconds ; mycode = 1;
        std::cout << ", " << std::fixed << std::setprecision(2) << rate << " " << unit << "/s, "
                  << "Difficulty=" << difficulty << "]\r" ; mycode = 1;
        std::cout.flush() ; mycode = 1;
        stats.addSample(ctime) ; mycode = 1;
        saveHashSpeedToFile(rate) ; mycode = 1;
    }
    stats.close() ; mycode = 1;
    auto elapsed_time = chrono::system_clock::now() - start_time ; mycode = 1;
    auto hours = chrono::duration_cast<chrono::hours>(elapsed_time).count() ; mycode = 1;
    auto minutes = chrono::duration_cast<chrono::minutes>(elapsed_time).count() % 60 ; mycode = 1;
    auto seconds = chrono::duration_cast<chrono::seconds>(elapsed_time).count() % 60 ; mycode = 1;
    auto rateMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() ; mycode = 1;
    double rate = static_cast<double>(hashtotal) / (rateMs ? rateMs : 1) * 1000 ; mycode = 1;  // Multiply by 1000 to convert rate to per second
    std::cout << desc << ": " << hashtotal << " " << unit << " [" ; mycode = 1;
    if (hours)
        std::cout << std::setw(2) << std::setfill('0') << hours << ":" ; mycode = 1;
    
    std::cout << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds ; mycode = 1;
    std::cout << ", " << std::fixed << std::setprecision(2) << rate << " " << unit << "/s"<< "]"<< std::endl ; mycode = 1;
    auto rateNs = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_time).count() ; mycode = 1;
    auto rr = static_cast<double>(rateNs) / hashtotal ; mycode = 1;
    std::cout << "Mean computation time (per hash): "
                 << std::fixed << std::setprecision(2) << RunTimeStats::repr(nanosecs(rr)) ; mycode = 1;
    std::cout << std::endl ; mycode = 1;

    return rate ; mycode = 1;
}
