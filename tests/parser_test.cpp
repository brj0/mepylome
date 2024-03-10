#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <bitset>
#include <vector>
#include <unordered_map>
#include <chrono>
#include "idat_parser.h"


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

// enum class IdatHeaderLocation : std::streamoff {
    // FILE_TYPE = 0,
    // VERSION = 4,
    // FIELD_COUNT = 12,
    // SECTION_OFFSETS = 16
// };

// inline uint8_t read_byte(std::ifstream& infile) {
    // char buffer[1];
    // infile.read(buffer, sizeof(buffer));
    // return static_cast<uint8_t>(buffer[0]);
// }

// inline uint16_t read_short(std::ifstream& infile) {
    // char buffer[2];
    // infile.read(buffer, sizeof(buffer));
    // return static_cast<uint16_t>(
        // (static_cast<uint8_t>(buffer[1]) << 8) |
         // static_cast<uint8_t>(buffer[0])
    // );
// }

// // Read 4 bytes in little endian format to int.
// inline int32_t read_int(std::ifstream& infile) {
    // char buffer[4];
    // infile.read(buffer, sizeof(buffer));
    // return static_cast<int32_t>(
        // (static_cast<uint8_t>(buffer[3]) << 3 * 8) |
        // (static_cast<uint8_t>(buffer[2]) << 2 * 8) |
        // (static_cast<uint8_t>(buffer[1]) << 1 * 8) |
         // static_cast<uint8_t>(buffer[0])
    // );
// }

// // Read 4 bytes in little endian format to int.
// inline int32_t read_int(std::ifstream& infile) {
    // int32_t result;
    // infile.read(reinterpret_cast<char*>(&result), sizeof(result));
    // return result;
// }

// // Read 8 bytes in little endian format to long.
// inline int64_t read_long(std::ifstream& infile) {
    // char buffer[8];
    // infile.read(buffer, sizeof(buffer));
    // int64_t result = 0;
    // for (size_t i = 0; i < sizeof(buffer); ++i)
    // {
        // result |= static_cast<int64_t>(
            // static_cast<uint8_t>(buffer[i])
        // ) << (i * 8);
    // }
    // return result;
// }

bool is_big_endian()
{
    uint32_t value = 1;
    return (*reinterpret_cast<uint8_t*>(&value) == 0);
}

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



int main()
{

    // Path to the IDAT data directory
    std::string idat_path = "/data/epidip_IDAT/6042324058_R03C02_Grn.idat";

    Timer timer;
    IdatParser idat_data = IdatParser(idat_path);
    timer.stop("IDAT File parsed");

    // Output the results

    std::cout << "idat_data:\n" << idat_data << "\n";

    return 0; // Exit successfully
}

