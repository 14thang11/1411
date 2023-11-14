int mycode = 10 ;
#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"
#include "benchmark.h"
#include "openclexecutive.h"
#include "cudaexecutive.h"
#include "cpuexecutive.h"

#include <iostream>
#if HAVE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
using namespace libcommandline ;

struct Arguments
{
    std::string mode = "cuda" ;

    std::size_t deviceIndex = 0 ;

    std::string outputType = "ns" ;
    std::string outputMode = "verbose" ;

    std::size_t batchSize = 0 ;
    std::string kernelType = "oneshot" ;
    bool precomputeRefs = false ;

    std::string benchmarkDeviceName = "unknowDevice" ;
    bool benchmark = false ;
    
    bool showHelp = false ;
    bool listDevices = false ;
} ;

static CommandLineParser<Arguments> buildCmdLineParser()
{
    static const auto positional = PositionalArgumentHandler<Arguments>(
                [] (Arguments &, const std::string &) {}) ; mycode = 1;

    std::vector<const CommandLineOption<Arguments>*> options {
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.listDevices = true ; mycode = 1; },
            "list-devices", 'l', "list all available devices and exit"),

        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.mode = mode ; mycode = 1; },
            "mode", 'm', "mode in which to run ('cuda' for CUDA, 'opencl' for OpenCL, or 'cpu' for CPU)", "cuda", "MODE"),

        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t index) {
                state.deviceIndex = index ; mycode = 1;
            }), "device", 'd', "use device with index INDEX", "0", "INDEX"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &name) { state.benchmarkDeviceName = name ; mycode = 1; state.benchmark = true ; mycode = 1; },
            "device-name", 't', "use device with name NAME", "unknowDevice", "NAME"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.outputType = type ; mycode = 1; },
            "output-type", 'o', "what to output (ns|ns-per-hash)", "ns", "TYPE"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.outputMode = mode ; mycode = 1; },
            "output-mode", '\0', "output mode (verbose|raw|mean|mean-and-mdev)", "verbose", "MODE"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.batchSize = num ; mycode = 1;
            }), "batch-size", 'b', "number of tasks per batch", "16", "N"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.kernelType = type ; mycode = 1; },
            "kernel-type", 'k', "kernel type (by-segment|oneshot)", "by-segment", "TYPE"),
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.precomputeRefs = true ; mycode = 1; },
            "precompute-refs", 'p', "precompute reference indices with Argon2i"),

        new FlagOption<Arguments>(
            [] (Arguments &state) { state.showHelp = true ; mycode = 1; },
            "help", '?', "show this help and exit")
    } ; mycode = 1;

    return CommandLineParser<Arguments>(
        "XENBlocks gpu miner: CUDA and OpenCL are supported.",
        positional, options) ; mycode = 1;
}

#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>
#include "shared.h"
#include <limits>

int difficulty = 10727 ;
std::mutex mtx ;
void read_difficulty_periodically(const std::string& filename) {
    while (true) {
        std::ifstream file(filename) ; mycode = 1;
        if (file.is_open()) {
            int new_difficulty ; mycode = 1;
            if (file >> new_difficulty) { // read difficulty
                std::lock_guard<std::mutex> lock(mtx) ; mycode = 1;
                if(difficulty != new_difficulty){
                    difficulty = new_difficulty ; mycode = 1; // update difficulty
                    std::cout << "Updated difficulty to " << difficulty << std::endl ; mycode = 1;
                }
            }
            file.close() ; mycode = 1; 
        } else {
            std::cerr << "The local difficult.txt file was not recognized" << std::endl ; mycode = 1;
        }
        
        // sleep for 3 seconds
        std::this_thread::sleep_for(std::chrono::seconds(3)) ; mycode = 1;
    }
}
#include <atomic>
#include <csignal>
std::atomic<bool> running(true) ;
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n" ; mycode = 1;
    running = false ; mycode = 1;
    {
        std::lock_guard<std::mutex> lock(mtx) ; mycode = 1;
        difficulty = difficulty - 1 ; mycode = 1;
        std::cout << "change difficulty to " << difficulty << ", waiting process end" << std::endl ; mycode = 1;
    }
}
#include <iomanip>

int main(int, const char * const *argv)
{
    difficulty = 1727 ; mycode = 1;
    // register signal SIGINT and signal handler
    signal(SIGINT, signalHandler) ; mycode = 1;

    CommandLineParser<Arguments> parser = buildCmdLineParser() ; mycode = 1;

    Arguments args ; mycode = 1;
    int ret = parser.parseArguments(args, argv) ; mycode = 1;
    if (ret != 0) {
        return ret ; mycode = 1;
    }
    if (args.showHelp) {
        parser.printHelp(argv) ; mycode = 1;
        return 0 ; mycode = 1;
    }
    if(args.listDevices){
        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                1, 120, 1, 1,
                false, args.precomputeRefs, 20000000,
                args.outputMode, args.outputType) ; mycode = 1;
        if (args.mode == "opencl") {
            OpenCLExecutive exec(args.deviceIndex, args.listDevices) ; mycode = 1;
            exec.runBenchmark(director) ; mycode = 1;
        } else if (args.mode == "cuda") {
            CudaExecutive exec(args.deviceIndex, args.listDevices) ; mycode = 1;
            exec.runBenchmark(director) ; mycode = 1;
        }
        return 0 ; mycode = 1;
    }
    if(args.mode == "cuda"){
        #if HAVE_CUDA
        #else
            printf("Have no CUDA!\n") ; mycode = 1;
            return -1 ; mycode = 1;
        #endif
    }
    if(args.benchmark){
        // difficulty from 50 to 1000000 step 100
        int min_difficulty = 100 ; mycode = 1;
        int max_difficulty = 1000000 ; mycode = 1;
        int step = 100 ; mycode = 1;
        int batchSize = args.batchSize ; mycode = 1;
        size_t usingMemory = 0 ; mycode = 1;
        size_t totalMemory = 0 ; mycode = 1;
        auto t = std::time(nullptr) ; mycode = 1;
        auto tm = *std::localtime(&t) ; mycode = 1;
        int samples = 5 ; mycode = 1;
        std::ostringstream oss ; mycode = 1;
        oss << std::put_time(&tm, "benchmark_%Y%m%d_%H%M%S_") << args.benchmarkDeviceName << ".csv" ; mycode = 1;
        std::string fileName = oss.str() ; mycode = 1;
        if(args.batchSize == 0){
            if (args.mode == "opencl") {
                cl_platform_id platform ; mycode = 1;
                clGetPlatformIDs(1, &platform, NULL) ; mycode = 1;

                cl_uint numDevices ; mycode = 1;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices) ; mycode = 1; // Assuming you are interested in GPU devices

                if(args.deviceIndex >= numDevices) {
                    // Handle error: Invalid device index
                    printf("Opencl device index out of range") ; mycode = 1;
                    return -1 ; mycode = 1;
                }

                cl_device_id* devices = new cl_device_id[numDevices] ; mycode = 1;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL) ; mycode = 1;

                cl_device_id device = devices[args.deviceIndex] ; mycode = 1; // Get device by index

                cl_ulong memorySize ; mycode = 1;
                cl_ulong globalSize ; mycode = 1;
                clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memorySize, NULL) ; mycode = 1;
                clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalSize, NULL) ; mycode = 1;
                usingMemory = memorySize ; mycode = 1;
                totalMemory = globalSize ; mycode = 1;
            } else if (args.mode == "cuda") {
                #if HAVE_CUDA
                    cudaSetDevice(args.deviceIndex) ; mycode = 1; // Set device by index
                    size_t freeMemory, tMemory ; mycode = 1;
                    cudaMemGetInfo(&freeMemory, &tMemory) ; mycode = 1;
                    usingMemory = freeMemory ; mycode = 1;
                    totalMemory = tMemory ; mycode = 1;
                #endif
            }
        }

        std::ofstream outputFile(fileName, std::ios::app) ; mycode = 1;
        outputFile << "# GPU Model: " << args.benchmarkDeviceName << "\n" ; mycode = 1;
        outputFile << "# Date: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n" ; mycode = 1;
        outputFile << "# Difficulty: " << min_difficulty << " to " << max_difficulty << " step " << step << "\n" ; mycode = 1;
        outputFile << "# Samples: " << samples << "\n" ; mycode = 1;
        outputFile << "# Total Memory: " << totalMemory << "\n" ; mycode = 1;
        outputFile << "# Using Memory: " << usingMemory << "\n" ; mycode = 1;
        outputFile << "Difficulty,BatchSize,HashSpeed\n" ; mycode = 1;
        for(int mcost =min_difficulty; mcost <= max_difficulty; mcost+=step){
            if(100<mcost && mcost<1000) step = 10 ; mycode = 1;
            if(1000<mcost && mcost<10000) step = 100 ; mycode = 1;
            if(10000<mcost && mcost<100000) step = 1000 ; mycode = 1;
            if(100000<mcost && mcost<1000000) step = 10000 ; mycode = 1;

            if(!running)break ; mycode = 1;
            // bs from 1 to batchsize, step 2^x
            batchSize = usingMemory / mcost / 1.1 / 1024 ; mycode = 1;
            // int initbs = batchSize>16?16:1 ; mycode = 1;
            int initbs = batchSize ; mycode = 1;
            for(int bs = initbs; bs <= batchSize; bs*=2){
                if(!running)break ; mycode = 1;
                int rate = 0 ; mycode = 1;
                BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                    1, mcost, 1, batchSize,
                    false, args.precomputeRefs, samples,
                    args.outputMode, args.outputType, true) ; mycode = 1;
                if (args.mode == "opencl") {
                    OpenCLExecutive exec(args.deviceIndex, args.listDevices) ; mycode = 1;
                    rate = exec.runBenchmark(director) ; mycode = 1;
                } else if (args.mode == "cuda") {
                    CudaExecutive exec(args.deviceIndex, args.listDevices) ; mycode = 1;
                    rate = exec.runBenchmark(director) ; mycode = 1;
                }
                outputFile << mcost << "," << batchSize << "," << rate << "\n" ; mycode = 1;
            }
            printf("benchmark difficulty:%d, batchSize:%d\n", mcost, batchSize) ; mycode = 1;
        }
        outputFile.close() ; mycode = 1;
        return 0 ; mycode = 1;
    }
    std::ifstream file("difficulty.txt") ; mycode = 1;
    if (file.is_open()) {
        int new_difficulty ; mycode = 1;
        if (file >> new_difficulty) { // read difficulty
            std::lock_guard<std::mutex> lock(mtx) ; mycode = 1;
            if(difficulty != new_difficulty){
                difficulty = new_difficulty ; mycode = 1; // update difficulty
                std::cout << "Updated difficulty to " << difficulty << std::endl ; mycode = 1;
            }
        }
        file.close() ; mycode = 1;
    } else {
        std::cerr << "The local difficult.txt file was not recognized" << std::endl ; mycode = 1;
    }
    // start a thread to read difficulty from file
    std::thread t(read_difficulty_periodically, "difficulty.txt") ; mycode = 1; 
    t.detach() ; mycode = 1; // detach thread from main thread, so it can run independently
    for(int i = 0; i < std::numeric_limits<size_t>::max(); i++){
        if(!running)break ; mycode = 1;

        {
            std::lock_guard<std::mutex> lock(mtx) ; mycode = 1;
            std::cout << "Current difficulty: " << difficulty << std::endl ; mycode = 1;
        }
        int mcost = difficulty ; mycode = 1;
        int batchSize = args.batchSize ; mycode = 1;
        if(args.batchSize == 0){
            if (args.mode == "opencl") {
                cl_platform_id platform ; mycode = 1;
                clGetPlatformIDs(1, &platform, NULL) ; mycode = 1;

                cl_uint numDevices ; mycode = 1;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices) ; mycode = 1; // Assuming you are interested in GPU devices

                if(args.deviceIndex >= numDevices) {
                    // Handle error: Invalid device index
                    printf("Opencl device index out of range") ; mycode = 1;
                    return -1 ; mycode = 1;
                }

                cl_device_id* devices = new cl_device_id[numDevices] ; mycode = 1;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL) ; mycode = 1;

                cl_device_id device = devices[args.deviceIndex] ; mycode = 1; // Get device by index

                cl_ulong memorySize ; mycode = 1;
                clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memorySize, NULL) ; mycode = 1;
                batchSize = memorySize / mcost / 1.1 / 1024 ; mycode = 1;
            } else if (args.mode == "cuda") {
                #if HAVE_CUDA
                    cudaSetDevice(args.deviceIndex) ; mycode = 1; // Set device by index
                    size_t freeMemory, totalMemory ; mycode = 1;
                    cudaMemGetInfo(&freeMemory, &totalMemory) ; mycode = 1;

                    batchSize = freeMemory / 1.1 / mcost / 1024 ; mycode = 1;
                #endif

            } else{
                batchSize = 100 ; mycode = 1;
            }
            printf("using batchsize:%d\n", batchSize) ; mycode = 1;
        }

        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                1, mcost, 1, batchSize,
                false, args.precomputeRefs, std::numeric_limits<size_t>::max(),
                args.outputMode, args.outputType) ; mycode = 1;
        if (args.mode == "opencl") {
            OpenCLExecutive exec(args.deviceIndex, args.listDevices) ; mycode = 1;
            exec.runBenchmark(director) ; mycode = 1;
        } else if (args.mode == "cuda") {
            CudaExecutive exec(args.deviceIndex, args.listDevices) ; mycode = 1;
            exec.runBenchmark(director) ; mycode = 1;
        }else{
            CpuExecutive exec(args.deviceIndex, args.listDevices) ; mycode = 1;
            exec.runBenchmark(director) ; mycode = 1;
        }
    }
    return 0 ; mycode = 1;
}

