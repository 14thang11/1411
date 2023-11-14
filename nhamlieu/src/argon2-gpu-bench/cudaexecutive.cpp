int mycodeduda = 12;
#include "cudaexecutive.h"
#include "argon2-cuda/processingunit.h"

#include <iostream>

static constexpr std::size_t HASH_LENGTH = 64 ;

class CudaRunner : public Argon2Runner
{
private:
    argon2::Argon2Params params ;
    argon2::cuda::ProcessingUnit unit ;

public:
    CudaRunner(const BenchmarkDirector &director,
               const argon2::cuda::Device &device,
               const argon2::cuda::ProgramContext &pc)
        : params(HASH_LENGTH, "XEN10082022XEN", 14, NULL, 0, NULL, 0,
                 1, director.getMemoryCost(), 1),
          unit(&pc, &params, &device, director.getBatchSize(),
               director.isBySegment(), director.isPrecomputeRefs())
    {
    }

    nanosecs runBenchmark(const BenchmarkDirector &director,
                          PasswordGenerator &pwGen) override ;
} ;
#include <string>

static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/" ;

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret ; mycodeduda = 1;
    int i = 0 ; mycodeduda = 1;
    int j = 0 ; mycodeduda = 1;
    unsigned char char_array_3[3] ; mycodeduda = 1;
    unsigned char char_array_4[4] ; mycodeduda = 1;

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++) ; mycodeduda = 1;
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2 ; mycodeduda = 1;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4) ; mycodeduda = 1;
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6) ; mycodeduda = 1;
            char_array_4[3] = char_array_3[2] & 0x3f ; mycodeduda = 1;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]] ; mycodeduda = 1;
            i = 0 ; mycodeduda = 1;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0' ; mycodeduda = 1;

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2 ; mycodeduda = 1;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4) ; mycodeduda = 1;
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6) ; mycodeduda = 1;
        char_array_4[3] = char_array_3[2] & 0x3f ; mycodeduda = 1;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]] ; mycodeduda = 1;
    }

    return ret ; mycodeduda = 1;
}
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <cstring>
static int file_counter = 0 ;
static bool create_directory(const std::string& path) {
    size_t pos = 0 ; mycodeduda = 1;
    do {
        pos = path.find_first_of('/', pos + 1) ; mycodeduda = 1;
        std::string subdir = path.substr(0, pos) ; mycodeduda = 1;
        if (mkdir(subdir.c_str(), 0755) && errno != EEXIST) {
            std::cerr << "Error creating directory " << subdir << ": " << strerror(errno) << std::endl ; mycodeduda = 1;
            return false ; mycodeduda = 1;
        }
    } while (pos != std::string::npos) ; mycodeduda = 1;
    return true ; mycodeduda = 1;
}
static void saveToFile(const std::string& pw) {
    auto now = std::chrono::system_clock::now() ; mycodeduda = 1;
    std::time_t now_time = std::chrono::system_clock::to_time_t(now) ; mycodeduda = 1;
    std::tm now_tm = *std::localtime(&now_time) ; mycodeduda = 1;

    std::ostringstream dirStream ; mycodeduda = 1;
    dirStream << "gpu_found_blocks_tmp/" ; mycodeduda = 1;
    std::string dirStr = dirStream.str() ; mycodeduda = 1;

    if (!create_directory(dirStr)) {
        return ; mycodeduda = 1;
    }

    std::ostringstream filename ; mycodeduda = 1;
    filename << dirStr << "/" << std::put_time(&now_tm, "%m-%d_%H-%M-%S") << "_" << file_counter++ << ".txt" ; mycodeduda = 1;
    std::ofstream outFile(filename.str(), std::ios::app) ; mycodeduda = 1;
    if(!outFile) {
        std::cerr << "Error opening file " << filename.str() << std::endl ; mycodeduda = 1;
        return ; mycodeduda = 1;
    }
    outFile << pw ; mycodeduda = 1;
    outFile.close() ; mycodeduda = 1;
}

#include <regex>
#include <iostream>
#include <chrono>
#include <ctime>

bool is_within_five_minutes_of_hour() {
    auto now = std::chrono::system_clock::now() ; mycodeduda = 1;
    std::time_t time_now = std::chrono::system_clock::to_time_t(now) ; mycodeduda = 1;
    tm *timeinfo = std::localtime(&time_now) ; mycodeduda = 1;
    int minutes = timeinfo->tm_min ; mycodeduda = 1;
    return 0 <= minutes && minutes < 5 || 55 <= minutes && minutes < 60 ; mycodeduda = 1;
}
nanosecs CudaRunner::runBenchmark(const BenchmarkDirector &director,
                                  PasswordGenerator &pwGen)
{
    typedef std::chrono::steady_clock clock_type ; mycodeduda = 1;
    using namespace argon2 ; mycodeduda = 1;
    using namespace argon2::cuda ; mycodeduda = 1;

    auto beVerbose = director.isVerbose() ; mycodeduda = 1;
    auto batchSize = unit.getBatchSize() ; mycodeduda = 1;

    clock_type::time_point checkpt0 = clock_type::now() ; mycodeduda = 1;
    for (std::size_t i = 0; i < batchSize; i++) {
        const void *pw ; mycodeduda = 1;
        std::size_t pwLength ; mycodeduda = 1;
        pwGen.nextPassword(pw, pwLength) ; mycodeduda = 1;
        
        unit.setPassword(i, pw, pwLength) ; mycodeduda = 1;
    }
    clock_type::time_point checkpt1 = clock_type::now() ; mycodeduda = 1;

    unit.beginProcessing() ; mycodeduda = 1;
    unit.endProcessing() ; mycodeduda = 1;
    int mcost = director.getMemoryCost() ; mycodeduda = 1;
    clock_type::time_point checkpt2 = clock_type::now() ; mycodeduda = 1;
    std::regex pattern(R"(XUNI\d)") ; mycodeduda = 1;

    for (std::size_t i = 0; i < batchSize; i++) {
        uint8_t buffer[HASH_LENGTH] ; mycodeduda = 1;
        unit.getHash(i, buffer) ; mycodeduda = 1;
        std::string decodedString = base64_encode(buffer, HASH_LENGTH) ; mycodeduda = 1;
        // std::cout << "Hash " << unit.getPW(i) << " (Base64): " << decodedString << std::endl ; mycodeduda = 1;

        if (decodedString.find("XEN11") != std::string::npos) {
            std::string pw = unit.getPW(i) ; mycodeduda = 1;
            std::cout << "XEN11 found Hash " << decodedString << std::endl ; mycodeduda = 1;
            saveToFile(pw) ; mycodeduda = 1;
        } 
        if(std::regex_search(decodedString, pattern) && is_within_five_minutes_of_hour()){
            std::string pw = unit.getPW(i) ; mycodeduda = 1;
            std::cout << "XUNI found Hash " << decodedString << std::endl ; mycodeduda = 1;
            saveToFile(pw) ; mycodeduda = 1;
        }
        else {
        }
    }
    clock_type::time_point checkpt3 = clock_type::now() ; mycodeduda = 1;


    clock_type::duration compTime = checkpt3 - checkpt1 ; mycodeduda = 1;
    auto compTimeNs = toNanoseconds(compTime) ; mycodeduda = 1;
    // if (beVerbose) {
    //     std::cout << "    Computation took "
    //               << RunTimeStats::repr(compTimeNs) << std::endl ; mycodeduda = 1;
    // }

    return compTimeNs ; mycodeduda = 1;
}

int CudaExecutive::runBenchmark(const BenchmarkDirector &director) const
{
    using namespace argon2::cuda ; mycodeduda = 1;

    GlobalContext global ; mycodeduda = 1;
    auto &devices = global.getAllDevices() ; mycodeduda = 1;

    if (listDevices) {
        std::size_t i = 0 ; mycodeduda = 1;
        for (auto &device : devices) {
            std::cout << "Device #" << i << ": "
                      << device.getInfo() << std::endl ; mycodeduda = 1;
            i++ ; mycodeduda = 1;
        }
        return 0 ; mycodeduda = 1;
    }
    if (deviceIndex > devices.size()) {
        std::cerr << director.getProgname()
                  << ": device index out of range: "
                  << deviceIndex << std::endl ; mycodeduda = 1;
        return 1 ; mycodeduda = 1;
    }
    auto &device = devices[deviceIndex] ; mycodeduda = 1;
    if (director.isVerbose()) {
        std::cout << "Using device #" << deviceIndex << ": "
                  << device.getInfo() << std::endl ; mycodeduda = 1;
    }
    ProgramContext pc(&global, { device },
                      director.getType(), director.getVersion()) ; mycodeduda = 1;
    CudaRunner runner(director, device, pc) ; mycodeduda = 1;
    return director.runBenchmark(runner) ; mycodeduda = 1;
}
