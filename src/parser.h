#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <map>


// Constants
constexpr int DEFAULT_IDAT_VERSION = 3;
constexpr const char* DEFAULT_IDAT_FILE_ID = "IDAT";


enum class IdatSectionCode : int
{
    ILLUMINA_ID = 102,
    STD_DEV = 103,
    MEAN = 104,
    NUM_BEADS = 107, // how many replicate measurements for each probe
    MID_BLOCK = 200,
    RUN_INFO = 300,
    RED_GREEN = 400,
    MOSTLY_NULL = 401, // manifest
    BARCODE = 402,
    CHIP_TYPE = 403, // format
    MOSTLY_A = 404,   // label
    UNKNOWN_1 = 405,  // opa
    UNKNOWN_2 = 406,  // sampleid
    UNKNOWN_3 = 407,  // descr
    UNKNOWN_4 = 408,  // plate
    UNKNOWN_5 = 409,  // well
    UNKNOWN_6 = 410,
    UNKNOWN_7 = 510,  // unknown
    NUM_SNPS_READ = 1000
};


std::ostream& operator<<(std::ostream& os, IdatSectionCode code);

template <typename T>
inline T read(std::ifstream& infile)
{
    T result;
    infile.read(reinterpret_cast<char*>(&result), sizeof(T));

    // Swap bytes if the machine is big-endian
    // if (is_big_endian())
    // {
        // std::reverse(
            // reinterpret_cast<char*>(&result),
            // reinterpret_cast<char*>(&result) + sizeof(T)
        // );
    // }

    return result;
}

template <typename T>
inline std::vector<T> read(std::ifstream& infile, const int num)
{
    std::vector<T> result(num);
    infile.read(reinterpret_cast<char*>(result.data()), sizeof(T) * num);
    return result;
}


inline uint8_t read_byte(std::ifstream& infile)
{
    return read<uint8_t>(infile);
}

inline uint16_t read_short(std::ifstream& infile)
{
    return read<uint16_t>(infile);
}

inline std::vector<uint16_t> read_short(std::ifstream& infile, const int num)
{
    return read<uint16_t>(infile, num);
}

inline int32_t read_int(std::ifstream& infile)
{
    return read<int32_t>(infile);
}

inline std::vector<int32_t> read_int(std::ifstream& infile, const int num)
{
    return read<int32_t>(infile, num);
}

inline int64_t read_long(std::ifstream& infile)
{
    return read<int64_t>(infile);
}

inline std::string read_char(std::ifstream& infile, const int num)
{
    char* buffer = new char[num];
    infile.read(buffer, num);
    std::string result(buffer, num);
    delete[] buffer;
    return result;
}

inline std::string read_string(std::ifstream& infile)
{
    int num = read_byte(infile);

    int num_chars = num % 128;
    int shift = 0;

    while (num / 128 == 1)
    {
        num = read_byte(infile);
        shift += 7;
        int offset = (num % 128) * (1 << shift);
        num_chars += offset;
    }

    return read_char(infile, num_chars);
}

template <typename T>
std::vector<T> read_array(std::istream& ifstream, std::size_t length)
{
    std::vector<char> buffer(sizeof(T) * length);
    ifstream.read(buffer.data(), buffer.size());
    if (ifstream.gcount() != static_cast<std::streamsize>(sizeof(T) * length))
    {
        throw std::ios_base::failure(
            "End of file reached before number of results parsed."
        );
    }
    return std::vector<T>(
        reinterpret_cast<T*>(buffer.data()),
        reinterpret_cast<T*>(buffer.data() + buffer.size())
    );
}

class IdatData
{
public:
    size_t file_size_;
    uint32_t num_fields_;
    std::unordered_map<IdatSectionCode, std::streamoff> offsets_;
    int32_t n_snps_read_;
    std::vector<int32_t> illumina_ids_;
    std::vector<uint16_t> std_dev_;
    std::vector<int16_t> probe_means_;
    std::vector<uint8_t> n_beads_;
    std::vector<int32_t> mid_block_;
    std::vector<std::string> run_info_;
    int32_t red_green_;
    std::string mostly_null_;
    std::string barcode_;
    std::string chip_type_;
    std::string mostly_a_;
    std::string unknown_1_;
    std::string unknown_2_;
    std::string unknown_3_;
    std::string unknown_4_;
    std::string unknown_5_;
    std::string unknown_6_;
    std::string unknown_7_;


    IdatData(const std::string& filepath);

    size_t get_file_size() const { return file_size_; }
    uint32_t get_num_fields() const { return num_fields_; }
    const std::unordered_map<
        IdatSectionCode, std::streamoff
    >& get_offsets() const { return offsets_; }
    int32_t get_n_snps_read() const { return n_snps_read_; }
    const std::vector<int32_t>& get_illumina_ids() const { return illumina_ids_; }
    const std::vector<uint16_t>& get_std_dev() const { return std_dev_; }
    const std::vector<int16_t>& get_probe_means() const { return probe_means_; }
    const std::vector<uint8_t>& get_n_beads() const { return n_beads_; }
    const std::vector<int32_t>& get_mid_block() const { return mid_block_; }
    const std::vector<std::string>& get_run_info() const { return run_info_; }
    int32_t get_red_green() const { return red_green_; }
    const std::string& get_mostly_null() const { return mostly_null_; }
    const std::string& get_barcode() const { return barcode_; }
    const std::string& get_chip_type() const { return chip_type_; }
    const std::string& get_mostly_a() const { return mostly_a_; }
    const std::string& get_unknown_1() const { return unknown_1_; }
    const std::string& get_unknown_2() const { return unknown_2_; }
    const std::string& get_unknown_3() const { return unknown_3_; }
    const std::string& get_unknown_4() const { return unknown_4_; }
    const std::string& get_unknown_5() const { return unknown_5_; }
    const std::string& get_unknown_6() const { return unknown_6_; }
    const std::string& get_unknown_7() const { return unknown_7_; }

    friend std::ostream& operator<<(std::ostream& os, const IdatData& data);

};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "array([";
    if (!vec.empty())
    {
        os << static_cast<int>(vec[0]);
        for (std::size_t i = 1; i < vec.size(); ++i)
        {
            if ((i < 3) || (i > vec.size() - 4) || (vec.size() < 10))
            {
                os << ", " << static_cast<int>(vec[i]);
            }
            else if (i == 3)
            {
                os << ", ...";
            }
        }
    }
    os << "], dtype=" << typeid(T).name() << ")";
    return os;
}

class Timer
{
    private:

        std::chrono::time_point<std::chrono::system_clock> time;

    public:

        Timer() { start(); }

        /*
         * @brief Start the timer.
         *
         * This function starts the timer and sets the initial timestamp.
         */
        void start()
        {
            time = std::chrono::high_resolution_clock::now();
        }

        /*
         * @brief Stop the timer and log a message.
         *
         * This function stops the timer, updates the end timestamp, and
         * logs a message.
         *
         * @param text The message to be logged.
         */
        void stop(const std::string& text)
        {
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Time passed: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - time
                   ).count()
                << " ms ("
                << text
                << ")\n";
            this->start();
        }

};
