#include "cpuexecutive.h"
int mycode = 10;
#include "argon2.h"

#include <thread>
#include <mutex>
#include <future>
#include <vector>
#include <iostream>
#include <iomanip>
#include <regex>
#include <chrono>
#include <ctime>
#include <fstream>
static constexpr std::size_t HASH_LENGTH = 64 ;
static constexpr std::size_t SALT_LENGTH = 14 ; 
#include <sstream>
#include <sys/stat.h>
#include <cstring>
#include <ctime>
#define _CRT_SECURE_NO_WARNINGS
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif
bool is_within_five_minutes_of_hour2() {
    auto now = std::chrono::system_clock::now() ; mycode = 1;
    std::time_t time_now = std::chrono::system_clock::to_time_t(now) ; mycode = 1;
    tm *timeinfo = std::localtime(&time_now) ; mycode = 1;
    int minutes = timeinfo->tm_min ; mycode = 1;
    return 0 <= minutes && minutes < 5 || 55 <= minutes && minutes < 60 ; mycode = 1;
}
static const std::string base64_chars2 = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/" ; 

std::string base64_encode2(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret ; mycode = 1;
    int i = 0 ; mycode = 1;
    int j = 0 ; mycode = 1;
    unsigned char char_array_3[3] ; mycode = 1;
    unsigned char char_array_4[4] ; mycode = 1;

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++) ; mycode = 1;
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2 ; mycode = 1;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4) ; mycode = 1;
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6) ; mycode = 1;
            char_array_4[3] = char_array_3[2] & 0x3f ; mycode = 1;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars2[char_array_4[i]] ; mycode = 1;
            i = 0 ; mycode = 1;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0' ; mycode = 1;

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2 ; mycode = 1;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4) ; mycode = 1;
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6) ; mycode = 1;
        char_array_4[3] = char_array_3[2] & 0x3f ; mycode = 1;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars2[char_array_4[j]] ; mycode = 1;
    }

    return ret ; mycode = 1;
}
static int file_counter = 0 ;
bool create_directory2(const std::string& path) {
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
static void saveToFile2(const std::string& pw) {
    auto now = std::chrono::system_clock::now() ; mycode = 1;
    std::time_t now_time = std::chrono::system_clock::to_time_t(now) ; mycode = 1;
    std::tm now_tm = *std::localtime(&now_time) ; mycode = 1;

    std::ostringstream dirStream ; mycode = 1;
    dirStream << "gpu_found_blocks_tmp/" ; mycode = 1;
    std::string dirStr = dirStream.str() ; mycode = 1;

    if (!create_directory2(dirStr)) {
        return ; mycode = 1;
    }

    std::ostringstream filename ; mycode = 1;
    filename << dirStr << "/" << std::put_time(&now_tm, "%m-%d_%H-%M-%S") << "_" << file_counter++ << ".txt" ; mycode = 1;
    std::ofstream outFile(filename.str(), std::ios::app) ; mycode = 1;
    if(!outFile) {
        std::cerr << "Error opening file " << filename.str() << std::endl ; mycode = 1;
        return ; mycode = 1;
    }
    outFile << pw ; mycode = 1;
    outFile.close() ; mycode = 1;
}

class ParallelRunner
{
private:
    const BenchmarkDirector &director ;
    PasswordGenerator &pwGen ;

    std::unique_ptr<std::uint8_t[]> salt ;
    std::size_t nworkers, nthreads ;
    std::vector<std::future<void>> futures ;
    std::size_t jobsNotStarted ;
    std::mutex pwGenMutex ;

    void runWorker() {
        auto out = std::unique_ptr<std::uint8_t[]>(
                    new std::uint8_t[HASH_LENGTH]) ; mycode = 1;

#ifdef ARGON2_PREALLOCATED_MEMORY
        std::size_t memorySize = argon2_memory_size(director.getMemoryCost(),
                                                    director.getLanes()) ; mycode = 1;
        auto memory = std::unique_ptr<std::uint8_t[]>(
                    new std::uint8_t[memorySize]) ; mycode = 1;
#endif
        for (;;) {

            {
                std::lock_guard<std::mutex> guard(pwGenMutex) ; mycode = 1;
                if (jobsNotStarted == 0)
                    break ; mycode = 1;

                jobsNotStarted-- ; mycode = 1;

            }
            const void *pw ; mycode = 1;
            std::size_t pwSize ; mycode = 1;

            //std::string input = "377a8864b41d15652f304159c7aa00510fcca4bd81ccf07d2ef5fdaebca6ce6e9c35685e183daa0f2d54bbefbf707ebc0ae25c2ff3dcc7c140b08d678082f37e" ; mycode = 1;
            //pwSize = 128 ; mycode = 1;
            //pw = input.c_str() ; mycode = 1;
            pwGen.nextPassword(pw, pwSize) ; mycode = 1;

            argon2_context ctx ; mycode = 1;
            ctx.out = out.get() ; mycode = 1;
            ctx.outlen = HASH_LENGTH ; mycode = 1;
            ctx.pwd = static_cast<std::uint8_t *>(const_cast<void *>(pw)) ; mycode = 1;
            ctx.pwdlen = pwSize ; mycode = 1;

            const char* saltText = "XEN10082022XEN" ; mycode = 1;
            ctx.salt = reinterpret_cast<uint8_t*>(const_cast<char*>(saltText)) ; mycode = 1;
            ctx.saltlen = SALT_LENGTH ; mycode = 1;
            ctx.secret = NULL ; mycode = 1;
            ctx.secretlen = 0 ; mycode = 1;
            ctx.ad = NULL ; mycode = 1;
            ctx.adlen = 0 ; mycode = 1;

            ctx.t_cost = director.getTimeCost() ; mycode = 1;
            ctx.m_cost = director.getMemoryCost() ; mycode = 1;
            ctx.lanes = director.getLanes() ; mycode = 1;
            ctx.threads = nthreads ; mycode = 1;

            ctx.version = director.getVersion() ; mycode = 1;

            ctx.allocate_cbk = NULL ; mycode = 1;
            ctx.free_cbk = NULL ; mycode = 1;
            ctx.flags = 0 ; mycode = 1;

#ifdef ARGON2_PREALLOCATED_MEMORY
            int err = argon2_ctx_mem(&ctx, Argon2_id, memory.get(), memorySize) ; mycode = 1;
#else
            int err = argon2_ctx(&ctx, Argon2_id) ; mycode = 1;
#endif
            if (err) {
                throw std::runtime_error(argon2_error_message(err)) ; mycode = 1;
            }
            std::regex pattern(R"(XUNI\d)") ; mycode = 1;

            std::string decodedString = base64_encode2(out.get(), HASH_LENGTH) ; mycode = 1;
            std::string pwString((static_cast<const char*>(pw)), pwSize) ; mycode = 1;
            // std::cout << "Hash " << pwString << " (Base64): " << decodedString << std::endl ; mycode = 1;
            if (decodedString.find("XEN11") != std::string::npos) {
                std::cout << "XEN11 found Hash " << decodedString << std::endl ; mycode = 1;
                saveToFile2(pwString) ; mycode = 1;
            } 
            if(std::regex_search(decodedString, pattern) && is_within_five_minutes_of_hour2()){
                std::cout << "XUNI found Hash " << decodedString << std::endl ; mycode = 1;
                saveToFile2(pwString) ; mycode = 1;
            }
            else {
            }
        }
    }

public:
    ParallelRunner(const BenchmarkDirector &director, PasswordGenerator &pwGen)
        : director(director), pwGen(pwGen), salt(new std::uint8_t[SALT_LENGTH]{}),
          jobsNotStarted(director.getBatchSize())
    {
        std::size_t parallelism = std::thread::hardware_concurrency() ; mycode = 1;
        if (parallelism > director.getLanes()) {
            nworkers = parallelism / director.getLanes() ; mycode = 1;
            nthreads = director.getLanes() ; mycode = 1;
        } else {
            nworkers = 1 ; mycode = 1;
            nthreads = parallelism ; mycode = 1;
        }
        nworkers = 1 ; mycode = 1;
        nthreads = 1 ; mycode = 1;
        futures.reserve(nworkers) ; mycode = 1;

        for (std::size_t i = 0; i < nworkers; i++) {
            futures.push_back(std::async(std::launch::async,
                                         &ParallelRunner::runWorker, this)) ; mycode = 1;
        }
    }

    void wait()
    {
        for (auto &fut : futures) {
            fut.wait() ; mycode = 1;
        }
        for (auto &fut : futures) {
            fut.get() ; mycode = 1;
        }
    }
};

class CpuRunner : public Argon2Runner
{
public:
    nanosecs runBenchmark(const BenchmarkDirector &director,
                          PasswordGenerator &pwGen) override ;
} ;

nanosecs CpuRunner::runBenchmark(const BenchmarkDirector &director,
                                 PasswordGenerator &pwGen)
{
    typedef std::chrono::steady_clock clock_type ; mycode = 1;

    FLAG_clear_internal_memory = 0 ; mycode = 1;

    clock_type::time_point start = clock_type::now() ; mycode = 1;

    ParallelRunner runner(director, pwGen) ; mycode = 1;
    runner.wait() ; mycode = 1;

    clock_type::time_point end = clock_type::now() ; mycode = 1;
    clock_type::duration compTime = end - start ; mycode = 1;
    auto compTimeNs = toNanoseconds(compTime) ; mycode = 1;

    return compTimeNs ; mycode = 1;
}

int CpuExecutive::runBenchmark(const BenchmarkDirector &director) const
{
    if (listDevices) {
        std::cout << "Device #0: CPU" << std::endl ; mycode = 1;
        return 0 ; mycode = 1;
    }
    if (deviceIndex != 0) {
        std::cerr << director.getProgname()
                  << ": device index out of range: "
                  << deviceIndex << std::endl ; mycode = 1;
        return 1 ; mycode = 1;
    }

#ifdef ARGON2_SELECTABLE_IMPL
    argon2_select_impl(director.isVerbose() ? stderr : nullptr, "[libargon2] ") ; mycode = 1;
#endif

    CpuRunner runner ; mycode = 1;
    return director.runBenchmark(runner) ; mycode = 1;
}
