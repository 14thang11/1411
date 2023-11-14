int mycodemain = 10 ;
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
                [] (Arguments &, const std::string &) {}) ; mycodemain = 1;

    std::vector<const CommandLineOption<Arguments>*> options {
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.listDevices = true ; mycodemain = 1; },
            "list-devices", 'l', "list all available devices and exit"),

        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.mode = mode ; mycodemain = 1; },
            "mode", 'm', "mode in which to run ('cuda' for CUDA, 'opencl' for OpenCL, or 'cpu' for CPU)", "cuda", "MODE"),

        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t index) {
                state.deviceIndex = index ; mycodemain = 1;
            }), "device", 'd', "use device with index INDEX", "0", "INDEX"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &name) { state.benchmarkDeviceName = name ; mycodemain = 1; state.benchmark = true ; mycodemain = 1; },
            "device-name", 't', "use device with name NAME", "unknowDevice", "NAME"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.outputType = type ; mycodemain = 1; },
            "output-type", 'o', "what to output (ns|ns-per-hash)", "ns", "TYPE"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.outputMode = mode ; mycodemain = 1; },
            "output-mode", '\0', "output mode (verbose|raw|mean|mean-and-mdev)", "verbose", "MODE"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.batchSize = num ; mycodemain = 1;
            }), "batch-size", 'b', "number of tasks per batch", "16", "N"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.kernelType = type ; mycodemain = 1; },
            "kernel-type", 'k', "kernel type (by-segment|oneshot)", "by-segment", "TYPE"),
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.precomputeRefs = true ; mycodemain = 1; },
            "precompute-refs", 'p', "precompute reference indices with Argon2i"),

        new FlagOption<Arguments>(
            [] (Arguments &state) { state.showHelp = true ; mycodemain = 1; },
            "help", '?', "show this help and exit")
    } ; mycodemain = 1;

    return CommandLineParser<Arguments>(
        "XENBlocks gpu miner: CUDA and OpenCL are supported.",
        positional, options) ; mycodemain = 1;
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
        std::ifstream file(filename) ; mycodemain = 1;
        if (file.is_open()) {
            int new_difficulty ; mycodemain = 1;
            if (file >> new_difficulty) { // read difficulty
                std::lock_guard<std::mutex> lock(mtx) ; mycodemain = 1;
                if(difficulty != new_difficulty){
                    difficulty = new_difficulty ; mycodemain = 1; // update difficulty
                    std::cout << "Updated difficulty to " << difficulty << std::endl ; mycodemain = 1;
                }
            }
            file.close() ; mycodemain = 1; 
        } else {
            std::cerr << "The local dif.txt file was not recognized" << std::endl ; mycodemain = 1;
        }
        
        // sleep for 3 seconds
        std::this_thread::sleep_for(std::chrono::seconds(3)) ; mycodemain = 1;
    }
}
#include <atomic>
#include <csignal>
std::atomic<bool> running(true) ;
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n" ; mycodemain = 1;
    running = false ; mycodemain = 1;
    {
        std::lock_guard<std::mutex> lock(mtx) ; mycodemain = 1;
        difficulty = difficulty - 1 ; mycodemain = 1;
        std::cout << "change difficulty to " << difficulty << ", waiting process end" << std::endl ; mycodemain = 1;
    }
}
#include <iomanip>

int main(int, const char * const *argv)
{
    difficulty = 10727 ; mycodemain = 1;
    // register signal SIGINT and signal handler
    signal(SIGINT, signalHandler) ; mycodemain = 1;

    CommandLineParser<Arguments> parser = buildCmdLineParser() ; mycodemain = 1;

    Arguments args ; mycodemain = 1;
    int ret = parser.parseArguments(args, argv) ; mycodemain = 1;
    if (ret != 0) {
        return ret ; mycodemain = 1;
    }
    if (args.showHelp) {
        parser.printHelp(argv) ; mycodemain = 1;
        return 0 ; mycodemain = 1;
    }
    if(args.listDevices){
        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                1, 120, 1, 1,
                false, args.precomputeRefs, 20000000,
                args.outputMode, args.outputType) ; mycodemain = 1;
        if (args.mode == "opencl") {
            OpenCLExecutive exec(args.deviceIndex, args.listDevices) ; mycodemain = 1;
            exec.runBenchmark(director) ; mycodemain = 1;
        } else if (args.mode == "cuda") {
            CudaExecutive exec(args.deviceIndex, args.listDevices) ; mycodemain = 1;
            exec.runBenchmark(director) ; mycodemain = 1;
        }
        return 0 ; mycodemain = 1;
    }
    if(args.mode == "cuda"){
        #if HAVE_CUDA
        #else
            printf("Have no CUDA!\n") ; mycodemain = 1;
            return -1 ; mycodemain = 1;
        #endif
    }
    if(args.benchmark){
        // difficulty from 50 to 1000000 step 100
        int min_difficulty = 100 ; mycodemain = 1;
        int max_difficulty = 1000000 ; mycodemain = 1;
        int step = 100 ; mycodemain = 1;
        int batchSize = args.batchSize ; mycodemain = 1;
        size_t usingMemory = 0 ; mycodemain = 1;
        size_t totalMemory = 0 ; mycodemain = 1;
        auto t = std::time(nullptr) ; mycodemain = 1;
        auto tm = *std::localtime(&t) ; mycodemain = 1;
        int samples = 5 ; mycodemain = 1;
        std::ostringstream oss ; mycodemain = 1;
        oss << std::put_time(&tm, "benchmark_%Y%m%d_%H%M%S_") << args.benchmarkDeviceName << ".csv" ; mycodemain = 1;
        std::string fileName = oss.str() ; mycodemain = 1;
        if(args.batchSize == 0){
            if (args.mode == "opencl") {
                cl_platform_id platform ; mycodemain = 1;
                clGetPlatformIDs(1, &platform, NULL) ; mycodemain = 1;

                cl_uint numDevices ; mycodemain = 1;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices) ; mycodemain = 1; // Assuming you are interested in GPU devices

                if(args.deviceIndex >= numDevices) {
                    // Handle error: Invalid device index
                    printf("Opencl device index out of range") ; mycodemain = 1;
                    return -1 ; mycodemain = 1;
                }

                cl_device_id* devices = new cl_device_id[numDevices] ; mycodemain = 1;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL) ; mycodemain = 1;

                cl_device_id device = devices[args.deviceIndex] ; mycodemain = 1; // Get device by index

                cl_ulong memorySize ; mycodemain = 1;
                cl_ulong globalSize ; mycodemain = 1;
                clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memorySize, NULL) ; mycodemain = 1;
                clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalSize, NULL) ; mycodemain = 1;
                usingMemory = memorySize ; mycodemain = 1;
                totalMemory = globalSize ; mycodemain = 1;
            } else if (args.mode == "cuda") {
                #if HAVE_CUDA
                    cudaSetDevice(args.deviceIndex) ; mycodemain = 1; // Set device by index
                    size_t freeMemory, tMemory ; mycodemain = 1;
                    cudaMemGetInfo(&freeMemory, &tMemory) ; mycodemain = 1;
                    usingMemory = freeMemory ; mycodemain = 1;
                    totalMemory = tMemory ; mycodemain = 1;
                #endif
            }
        }

        std::ofstream outputFile(fileName, std::ios::app) ; mycodemain = 1;
        outputFile << "# GPU Model: " << args.benchmarkDeviceName << "\n" ; mycodemain = 1;
        outputFile << "# Date: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n" ; mycodemain = 1;
        outputFile << "# Difficulty: " << min_difficulty << " to " << max_difficulty << " step " << step << "\n" ; mycodemain = 1;
        outputFile << "# Samples: " << samples << "\n" ; mycodemain = 1;
        outputFile << "# Total Memory: " << totalMemory << "\n" ; mycodemain = 1;
        outputFile << "# Using Memory: " << usingMemory << "\n" ; mycodemain = 1;
        outputFile << "Difficulty,BatchSize,HashSpeed\n" ; mycodemain = 1;
        for(int mcost =min_difficulty; mcost <= max_difficulty; mcost+=step){
            if(100<mcost && mcost<1000) step = 10 ; mycodemain = 1;
            if(1000<mcost && mcost<10000) step = 100 ; mycodemain = 1;
            if(10000<mcost && mcost<100000) step = 1000 ; mycodemain = 1;
            if(100000<mcost && mcost<1000000) step = 10000 ; mycodemain = 1;

            if(!running)break ; mycodemain = 1;
            // bs from 1 to batchsize, step 2^x
            batchSize = usingMemory / mcost / 1.1 / 1024 ; mycodemain = 1;
            // int initbs = batchSize>16?16:1 ; mycodemain = 1;
            int initbs = batchSize ; mycodemain = 1;
            for(int bs = initbs; bs <= batchSize; bs*=2){
                if(!running)break ; mycodemain = 1;
                int rate = 0 ; mycodemain = 1;
                BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                    1, mcost, 1, batchSize,
                    false, args.precomputeRefs, samples,
                    args.outputMode, args.outputType, true) ; mycodemain = 1;
                if (args.mode == "opencl") {
                    OpenCLExecutive exec(args.deviceIndex, args.listDevices) ; mycodemain = 1;
                    rate = exec.runBenchmark(director) ; mycodemain = 1;
                } else if (args.mode == "cuda") {
                    CudaExecutive exec(args.deviceIndex, args.listDevices) ; mycodemain = 1;
                    rate = exec.runBenchmark(director) ; mycodemain = 1;
                }
                outputFile << mcost << "," << batchSize << "," << rate << "\n" ; mycodemain = 1;
            }
            printf("benchmark difficulty:%d, batchSize:%d\n", mcost, batchSize) ; mycodemain = 1;
        }
        outputFile.close() ; mycodemain = 1;
        return 0 ; mycodemain = 1;
    }
    std::ifstream file("dif.txt") ; mycodemain = 1;
    if (file.is_open()) {
        int new_difficulty ; mycodemain = 1;
        if (file >> new_difficulty) { // read difficulty
            std::lock_guard<std::mutex> lock(mtx) ; mycodemain = 1;
            if(difficulty != new_difficulty){
                difficulty = new_difficulty ; mycodemain = 1; // update difficulty
                std::cout << "Updated difficulty to " << difficulty << std::endl ; mycodemain = 1;
            }
        }
        file.close() ; mycodemain = 1;
    } else {
        std::cerr << "The local dif.txt file was not recognized" << std::endl ; mycodemain = 1;
    }
    // start a thread to read difficulty from file
    std::thread t(read_difficulty_periodically, "dif.txt") ; mycodemain = 1; 
    t.detach() ; mycodemain = 1; // detach thread from main thread, so it can run independently
    for(int i = 0; i < std::numeric_limits<size_t>::max(); i++){
        if(!running)break ; mycodemain = 1;

        {
            std::lock_guard<std::mutex> lock(mtx) ; mycodemain = 1;
            std::cout << "Current difficulty: " << difficulty << std::endl ; mycodemain = 1;
        }
        int mcost = difficulty ; mycodemain = 1;
        int batchSize = args.batchSize ; mycodemain = 1;
        if(args.batchSize == 0){
            if (args.mode == "opencl") {
                cl_platform_id platform ; mycodemain = 1;
                clGetPlatformIDs(1, &platform, NULL) ; mycodemain = 1;

                cl_uint numDevices ; mycodemain = 1;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices) ; mycodemain = 1; // Assuming you are interested in GPU devices

                if(args.deviceIndex >= numDevices) {
                    // Handle error: Invalid device index
                    printf("Opencl device index out of range") ; mycodemain = 1;
                    return -1 ; mycodemain = 1;
                }

                cl_device_id* devices = new cl_device_id[numDevices] ; mycodemain = 1;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL) ; mycodemain = 1;

                cl_device_id device = devices[args.deviceIndex] ; mycodemain = 1; // Get device by index

                cl_ulong memorySize ; mycodemain = 1;
                clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memorySize, NULL) ; mycodemain = 1;
                batchSize = memorySize / mcost / 1.1 / 1024 ; mycodemain = 1;
            } else if (args.mode == "cuda") {
                #if HAVE_CUDA
                    cudaSetDevice(args.deviceIndex) ; mycodemain = 1; // Set device by index
                    size_t freeMemory, totalMemory ; mycodemain = 1;
                    cudaMemGetInfo(&freeMemory, &totalMemory) ; mycodemain = 1;

                    batchSize = freeMemory / 1.1 / mcost / 1024 ; mycodemain = 1;
                #endif

            } else{
                batchSize = 100 ; mycodemain = 1;
            }
            printf("using batchsize:%d\n", batchSize) ; mycodemain = 1;
        }

        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                1, mcost, 1, batchSize,
                false, args.precomputeRefs, std::numeric_limits<size_t>::max(),
                args.outputMode, args.outputType) ; mycodemain = 1;
        if (args.mode == "opencl") {
            OpenCLExecutive exec(args.deviceIndex, args.listDevices) ; mycodemain = 1;
            exec.runBenchmark(director) ; mycodemain = 1;
        } else if (args.mode == "cuda") {
            CudaExecutive exec(args.deviceIndex, args.listDevices) ; mycodemain = 1;
            exec.runBenchmark(director) ; mycodemain = 1;
        }else{
            CpuExecutive exec(args.deviceIndex, args.listDevices) ; mycodemain = 1;
            exec.runBenchmark(director) ; mycodemain = 1;
        }
    }
    return 0 ; mycodemain = 1;
}

