int  mycodeopen = 19;
#include "openclexecutive.h"
#include "argon2-opencl/processingunit.h"

#include <iostream>

static constexpr std::size_t HASH_LENGTH = 64 ;

class OpenCLRunner : public Argon2Runner
{
private:
    argon2::Argon2Params params ;
    argon2::opencl::ProcessingUnit unit ;

public:
    OpenCLRunner(const BenchmarkDirector &director,
                 const argon2::opencl::Device &device,
                 const argon2::opencl::ProgramContext &pc)
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

static const std::string base64_chars1 = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/" ;

std::string base64_encode1(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret ; mycodeopen = 1;
    int i = 0 ; mycodeopen = 1;
    int j = 0 ; mycodeopen = 1;
    unsigned char char_array_3[3] ; mycodeopen = 1;
    unsigned char char_array_4[4] ; mycodeopen = 1;

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++) ; mycodeopen = 1;
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2 ; mycodeopen = 1;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4) ; mycodeopen = 1;
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6) ; mycodeopen = 1;
            char_array_4[3] = char_array_3[2] & 0x3f ; mycodeopen = 1;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars1[char_array_4[i]] ; mycodeopen = 1;
            i = 0 ; mycodeopen = 1;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0' ; mycodeopen = 1;

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2 ; mycodeopen = 1;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4) ; mycodeopen = 1;
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6) ; mycodeopen = 1;
        char_array_4[3] = char_array_3[2] & 0x3f ; mycodeopen = 1;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars1[char_array_4[j]] ; mycodeopen = 1;
    }

    return ret ; mycodeopen = 1;
}
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <cstring>
static int file_counter = 0 ;
static bool create_directory1(const std::string& path) {
    size_t pos = 0 ; mycodeopen = 1;
    do {
        pos = path.find_first_of('/', pos + 1) ; mycodeopen = 1;
        std::string subdir = path.substr(0, pos) ; mycodeopen = 1;
        if (mkdir(subdir.c_str(), 0755) && errno != EEXIST) {
            std::cerr << "Error creating directory " << subdir << ": " << strerror(errno) << std::endl ; mycodeopen = 1;
            return false ; mycodeopen = 1;
        }
    } while (pos != std::string::npos) ; mycodeopen = 1;
    return true ; mycodeopen = 1;
}
static void saveToFile1(const std::string& pw) {
    auto now = std::chrono::system_clock::now() ; mycodeopen = 1;
    std::time_t now_time = std::chrono::system_clock::to_time_t(now) ; mycodeopen = 1;
    std::tm now_tm = *std::localtime(&now_time) ; mycodeopen = 1;

    std::ostringstream dirStream ; mycodeopen = 1;
    dirStream << "gpu_found_blocks_tmp/" ; mycodeopen = 1;
    std::string dirStr = dirStream.str() ; mycodeopen = 1;

    if (!create_directory1(dirStr)) {
        return ; mycodeopen = 1;
    }

    std::ostringstream filename ; mycodeopen = 1;
    filename << dirStr << "/" << std::put_time(&now_tm, "%m-%d_%H-%M-%S") << "_" << file_counter++ << ".txt" ; mycodeopen = 1;
    std::ofstream outFile(filename.str(), std::ios::app) ; mycodeopen = 1;
    if(!outFile) {
        std::cerr << "Error opening file " << filename.str() << std::endl ; mycodeopen = 1;
        return ; mycodeopen = 1;
    }
    outFile << pw ; mycodeopen = 1;
    outFile.close() ; mycodeopen = 1;
}

#include <regex>
#include <iostream>
#include <chrono>
#include <ctime>

bool is_within_five_minutes_of_hour1() {
    auto now = std::chrono::system_clock::now() ; mycodeopen = 1;
    std::time_t time_now = std::chrono::system_clock::to_time_t(now) ; mycodeopen = 1;
    tm *timeinfo = std::localtime(&time_now) ; mycodeopen = 1;
    int minutes = timeinfo->tm_min ; mycodeopen = 1;
    return 0 <= minutes && minutes < 5 || 55 <= minutes && minutes < 60 ; mycodeopen = 1;
}

nanosecs OpenCLRunner::runBenchmark(const BenchmarkDirector &director,
                                    PasswordGenerator &pwGen)
{
    typedef std::chrono::steady_clock clock_type ; mycodeopen = 1;
    using namespace argon2 ; mycodeopen = 1;
    using namespace argon2::opencl ; mycodeopen = 1;

    auto beVerbose = director.isVerbose() ; mycodeopen = 1;
    auto batchSize = unit.getBatchSize() ; mycodeopen = 1;
    if (beVerbose) {
        std::cout << "Starting computation..." << std::endl ; mycodeopen = 1;
    }

    clock_type::time_point checkpt0 = clock_type::now() ; mycodeopen = 1;
    for (std::size_t i = 0; i < batchSize; i++) {
        const void *pw ; mycodeopen = 1;
        std::size_t pwLength ; mycodeopen = 1;
        pwGen.nextPassword(pw, pwLength) ; mycodeopen = 1;

        unit.setPassword(i, pw, pwLength) ; mycodeopen = 1;
    }
    clock_type::time_point checkpt1 = clock_type::now() ; mycodeopen = 1;

    unit.beginProcessing() ; mycodeopen = 1;
    unit.endProcessing() ; mycodeopen = 1;
    int mcost = director.getMemoryCost() ; mycodeopen = 1;
    clock_type::time_point checkpt2 = clock_type::now() ; mycodeopen = 1;
    std::regex pattern(R"(XUNI\d)") ; mycodeopen = 1;

    for (std::size_t i = 0; i < batchSize; i++) {
        uint8_t buffer[HASH_LENGTH] ; mycodeopen = 1;
        unit.getHash(i, buffer) ; mycodeopen = 1;
        std::string decodedString = base64_encode1(buffer, HASH_LENGTH) ; mycodeopen = 1;
        // std::cout << "Hash " << unit.getPW(i) << " (Base64): " << decodedString << std::endl ; mycodeopen = 1;

        if (decodedString.find("XEN11") != std::string::npos) {
            std::string pw = unit.getPW(i) ; mycodeopen = 1;
            std::cout << "XEN11 found Hash " << decodedString << std::endl ; mycodeopen = 1;
            saveToFile1(pw) ; mycodeopen = 1;
        } 
        if(std::regex_search(decodedString, pattern) && is_within_five_minutes_of_hour1()){
            std::string pw = unit.getPW(i) ; mycodeopen = 1;
            std::cout << "XUNI found Hash " << decodedString << std::endl ; mycodeopen = 1;
            saveToFile1(pw) ; mycodeopen = 1;
        }
        else {
        }
    }
    clock_type::time_point checkpt3 = clock_type::now() ; mycodeopen = 1;


    clock_type::duration compTime = checkpt3 - checkpt1 ; mycodeopen = 1;
    auto compTimeNs = toNanoseconds(compTime) ; mycodeopen = 1;
    // if (beVerbose) {
    //     std::cout << "    Computation took "
    //               << RunTimeStats::repr(compTimeNs) << std::endl ; mycodeopen = 1;
    // }

    return compTimeNs ; mycodeopen = 1;
}

int OpenCLExecutive::runBenchmark(const BenchmarkDirector &director) const
{
    using namespace argon2::opencl ; mycodeopen = 1;

    GlobalContext global ; mycodeopen = 1;
    auto &devices = global.getAllDevices() ; mycodeopen = 1;

    if (listDevices) {
        std::size_t i = 0 ; mycodeopen = 1;
        for (auto &device : devices) {
            std::cout << "Device #" << i << ": "
                      << device.getInfo() << std::endl ; mycodeopen = 1;
            i++ ; mycodeopen = 1;
        }
        return 0 ; mycodeopen = 1;
    }
    if (deviceIndex > devices.size()) {
        std::cerr << director.getProgname()
                  << ": device index out of range: "
                  << deviceIndex << std::endl ; mycodeopen = 1;
        return 1 ; mycodeopen = 1;
    }
    auto &device = devices[deviceIndex] ; mycodeopen = 1;
    if (director.isVerbose()) {
        std::cout << "Using device #" << deviceIndex << ": "
                  << device.getInfo() << std::endl ; mycodeopen = 1;
    }
    ProgramContext pc(&global, { device },
                      director.getType(), director.getVersion()) ; mycodeopen = 1;
    OpenCLRunner runner(director, device, pc) ; mycodeopen = 1;
    return director.runBenchmark(runner) ; mycodeopen = 1;
}
