#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <bitset>
#include <vector>
#include <unordered_map>
#include <chrono>


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

std::ostream& operator<<(std::ostream& os, IdatSectionCode code)
{
    os << static_cast<int>(code);
    return os;
}

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

std::ifstream::pos_type filesize(const std::string filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}


int main()
{
    Timer timer;
    constexpr int DEFAULT_IDAT_VERSION = 3;
    constexpr const char* DEFAULT_IDAT_FILE_ID = "IDAT";

    // Path to the IDAT data directory
    std::string idat_path = (std::string)getenv("HOME") +
        "/MEGA/work/programming/pyidat/101130760092_R05C02_Grn.idat";

    int file_size = static_cast<int>(filesize(idat_path));

    std::ifstream idat_file(idat_path, std::ios::binary);

    if (!idat_file.is_open())
    {
        std::cerr << "Error opening file: " << idat_path << std::endl;
        return 1;
    }


    // Parse Header of IDAT file

    std::string file_type = read_char(idat_file, std::strlen(DEFAULT_IDAT_FILE_ID));

    if (file_type != DEFAULT_IDAT_FILE_ID)
    {
        std::cerr << "Not an IDAT file. Unsupported file type."
                  << std::endl;
        return 1;
    }

    uint64_t idat_version = read_long(idat_file);

    if (idat_version != DEFAULT_IDAT_VERSION)
    {
        std::cerr << "Not a version 3 IDAT file. Unsupported IDAT version."
                  << std::endl;
        return 1;
    }

    uint32_t num_fields = read_int(idat_file);

    std::unordered_map<IdatSectionCode, std::streamoff> offsets;
    for (uint32_t i = 0; i < num_fields; ++i)
    {
        uint16_t key = read_short(idat_file);
        int64_t offset = read_long(idat_file);
        offsets[static_cast<IdatSectionCode>(key)] = std::streamoff(offset);
    }


    // Parse Body of IDAT file

    idat_file.seekg(offsets[IdatSectionCode::NUM_SNPS_READ]);
    int32_t n_snps_read = read_int(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::ILLUMINA_ID]);
    std::vector<int32_t> illumina_ids = read_array<int32_t>(idat_file, n_snps_read);

    idat_file.seekg(offsets[IdatSectionCode::STD_DEV]);
    std::vector<uint16_t> std_dev = read_short(idat_file, n_snps_read);

    idat_file.seekg(offsets[IdatSectionCode::MEAN]);
    std::vector<int16_t> probe_means = read_array<int16_t>(idat_file, n_snps_read);

    idat_file.seekg(offsets[IdatSectionCode::NUM_BEADS]);
    std::vector<uint8_t> n_beads = read_array<uint8_t>(idat_file, n_snps_read);

    idat_file.seekg(offsets[IdatSectionCode::MID_BLOCK]);
    int32_t n_mid_block = read_int(idat_file);
    std::vector<int32_t> mid_block = read_int(idat_file, n_mid_block);

    idat_file.seekg(offsets[IdatSectionCode::RUN_INFO]);
    int32_t runinfo_entry_count = read_int(idat_file);
    std::vector<std::string> run_info(5*runinfo_entry_count + 5);
    run_info[0] = "run_time";
    run_info[1] = "block_type";
    run_info[2] = "block_pars";
    run_info[3] = "block_code";
    run_info[4] = "code_version";
    for (int i = 1; i < runinfo_entry_count + 1; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            run_info[i*5 + j] = read_string(idat_file);
        }
    }

    idat_file.seekg(offsets[IdatSectionCode::RED_GREEN]);
    int32_t red_green = read_int(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::MOSTLY_NULL]);
    std::string mostly_null = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::BARCODE]);
    std::string barcode = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::CHIP_TYPE]);
    std::string chip_type = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::MOSTLY_A]);
    std::string mostly_a = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::UNKNOWN_1]);
    std::string unknown_1 = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::UNKNOWN_2]);
    std::string unknown_2 = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::UNKNOWN_3]);
    std::string unknown_3 = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::UNKNOWN_4]);
    std::string unknown_4 = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::UNKNOWN_5]);
    std::string unknown_5 = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::UNKNOWN_6]);
    std::string unknown_6 = read_string(idat_file);

    idat_file.seekg(offsets[IdatSectionCode::UNKNOWN_7]);
    std::string unknown_7 = read_string(idat_file);


    idat_file.close();

    timer.stop("IDAT File parsed");



    // Output the results

    std::cout << "is_big_endian: " << is_big_endian() << "\n";
    std::cout << "file_size: " << file_size << std::endl;

    // IDAT Header
    std::cout << "File Type: " << file_type << std::endl;
    std::cout << "IDAT Version: " << idat_version << std::endl;
    std::cout << "Number of Fields: " << num_fields << std::endl;
    for (const auto& entry : offsets)
    {
        std::cout << "\tkey: " << entry.first << "\toffset: "
                  << entry.second << std::endl;
    }

    // IDAT Body
    std::cout << "n_snps_read: " << n_snps_read << std::endl;
    std::cout << "illumina_ids: " << illumina_ids << std::endl;
    std::cout << "std_dev: " << std_dev << std::endl;
    std::cout << "probe_means: " << probe_means << std::endl;
    std::cout << "n_beads: " << n_beads << std::endl;
    std::cout << "n_mid_block: " << n_mid_block << std::endl;
    std::cout << "mid_block: " << mid_block << std::endl;
    std::cout << "runinfo_entry_count: " << runinfo_entry_count << std::endl;
    std::cout << "run_info: \n";
    for (int i = 0; i < runinfo_entry_count + 1; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            std::cout << run_info[i*5 + j] << ", ";
        }
            std::cout << std::endl;
    }
    std::cout << "red_green: " << red_green << std::endl;
    std::cout << "mostly_null: " << mostly_null << std::endl;
    std::cout << "barcode: " << barcode << std::endl;
    std::cout << "chip_type: " << chip_type << std::endl;
    std::cout << "mostly_a: " << mostly_a << std::endl;
    std::cout << "unknown_1: " << unknown_1 << std::endl;
    std::cout << "unknown_2: " << unknown_2 << std::endl;
    std::cout << "unknown_3: " << unknown_3 << std::endl;
    std::cout << "unknown_4: " << unknown_4 << std::endl;
    std::cout << "unknown_5: " << unknown_5 << std::endl;
    std::cout << "unknown_6: " << unknown_6 << std::endl;
    std::cout << "unknown_7: " << unknown_7 << std::endl;

    return 0; // Exit successfully
}

