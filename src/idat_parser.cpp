

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "This code is not compatible with big-endian systems."
#endif

#include <sstream>
#include "idat_parser.h"



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

inline uint8_t read_byte(std::ifstream& infile)
{
    return read<uint8_t>(infile);
}

inline uint16_t read_short(std::ifstream& infile)
{
    return read<uint16_t>(infile);
}


inline int32_t read_int(std::ifstream& infile)
{
    return read<int32_t>(infile);
}

inline int64_t read_long(std::ifstream& infile)
{
    return read<int64_t>(infile);
}

inline std::string read_char(std::ifstream& infile, const int num)
{
    std::string result(num, '\0');
    infile.read(&result[0], num);
    return result;
}

inline std::string read_string(std::ifstream& infile)
{
    int num = read_byte(infile);

    int num_chars = num % 128;
    int shift = 0;

    while (num / 128 == 1)
    {
        if (infile.peek() == EOF) {
            throw std::ios_base::failure(
                "Parser reached the end of the IDAT prematurely (read_string)."
            );
        }
        num = read_byte(infile);
        shift += 7;
        int offset = (num % 128) * (1 << shift);
        num_chars += offset;
    }

    return read_char(infile, num_chars);
}

template <typename T>
std::vector<T> read_array(std::istream& infile, std::size_t length)
{
    std::vector<T> vec(length);
    infile.read(reinterpret_cast<char*>(vec.data()), length * sizeof(T));
    if (infile.gcount() != static_cast<std::streamsize>(length * sizeof(T)))
    {
        throw std::ios_base::failure(
            "Parser reached the end of the file prematurely (read_array)."
        );
    }
    return vec;
}

std::ostream& operator<<(std::ostream& os, IdatSectionCode code)
{
    os << static_cast<int>(code);
    return os;
}

std::ifstream::pos_type filesize(const std::string filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

IdatParser::IdatParser(
    const std::string& filepath,
    bool intensity_only,
    bool array_type_only
)
    : file_size_(static_cast<size_t>(filesize(filepath)))
    , intensity_only(intensity_only)
    , array_type_only(array_type_only)
{
    // Start parsing IDAT file

    std::ifstream idat_file(filepath, std::ios::binary);

    if (!idat_file.is_open())
    {
        throw std::ios_base::failure(
            "Parser could not open file " + filepath
        );
    }


    // Parse IDAT header

    std::string file_type = read_char(idat_file, std::strlen(DEFAULT_IDAT_FILE_ID));

    if (file_type != DEFAULT_IDAT_FILE_ID)
    {
        throw std::ios_base::failure(
            "Parser could not open file as its not a valid IDAT file."
        );
    }

    uint64_t idat_version = read_long(idat_file);

    if (idat_version != DEFAULT_IDAT_VERSION)
    {
        throw std::ios_base::failure(
            "Parser could not open file as its not a version 3 IDAT file."
        );
    }

    num_fields_ = read_int(idat_file);

    for (uint32_t i = 0; i < num_fields_; ++i)
    {
        uint16_t key = read_short(idat_file);
        int64_t offset = read_long(idat_file);
        offsets_[static_cast<IdatSectionCode>(key)] = std::streamoff(offset);
    }


    // Parse IDAT body

    idat_file.seekg(offsets_[IdatSectionCode::NUM_SNPS_READ]);
    n_snps_read_ = read_int(idat_file);

    if (array_type_only)
    {
        return;
    }

    idat_file.seekg(offsets_[IdatSectionCode::ILLUMINA_ID]);
    illumina_ids_ = read_array<int32_t>(idat_file, n_snps_read_);

    idat_file.seekg(offsets_[IdatSectionCode::MEAN]);
    probe_means_= read_array<uint16_t>(idat_file, n_snps_read_);

    if (intensity_only)
    {
        return;
    }

    idat_file.seekg(offsets_[IdatSectionCode::STD_DEV]);
    std_dev_= read_array<uint16_t>(idat_file, n_snps_read_);

    idat_file.seekg(offsets_[IdatSectionCode::NUM_BEADS]);
    n_beads_= read_array<uint8_t>(idat_file, n_snps_read_);

    idat_file.seekg(offsets_[IdatSectionCode::MID_BLOCK]);
    int32_t n_mid_block = read_int(idat_file);
    mid_block_ = read_array<int32_t>(idat_file, n_mid_block);

    idat_file.seekg(offsets_[IdatSectionCode::RUN_INFO]);
    int32_t runinfo_entry_count = read_int(idat_file);
    run_info_.resize(5*runinfo_entry_count + 5);
    run_info_[0] = "run_time";
    run_info_[1] = "block_type";
    run_info_[2] = "block_pars";
    run_info_[3] = "block_code";
    run_info_[4] = "code_version";
    for (int i = 1; i < runinfo_entry_count + 1; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            run_info_[i*5 + j] = read_string(idat_file);
        }
    }

    idat_file.seekg(offsets_[IdatSectionCode::RED_GREEN]);
    red_green_ = read_int(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::MOSTLY_NULL]);
    mostly_null_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::BARCODE]);
    barcode_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::CHIP_TYPE]);
    chip_type_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::MOSTLY_A]);
    mostly_a_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::UNKNOWN_1]);
    unknown_1_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::UNKNOWN_2]);
    unknown_2_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::UNKNOWN_3]);
    unknown_3_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::UNKNOWN_4]);
    unknown_4_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::UNKNOWN_5]);
    unknown_5_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::UNKNOWN_6]);
    unknown_6_ = read_string(idat_file);

    idat_file.seekg(offsets_[IdatSectionCode::UNKNOWN_7]);
    unknown_7_ = read_string(idat_file);


    idat_file.close();
}

// Default implementation returns the raw type name
template<typename T>
std::string type_name() {
    return typeid(T).name();
}

// Specialization for specific types
template<>
std::string type_name<int8_t>() {
    return "int8";
}

template<>
std::string type_name<uint8_t>() {
    return "uint8";
}

template<>
std::string type_name<int16_t>() {
    return "int16";
}

template<>
std::string type_name<uint16_t>() {
    return "uint16";
}

template<>
std::string type_name<int32_t>() {
    return "int32";
}

template<>
std::string type_name<uint32_t>() {
    return "uint32";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    size_t n_visable = 2;
    os << "array([";
    if (!vec.empty())
    {
        os << static_cast<int>(vec[0]);
        for (std::size_t i = 1; i < vec.size(); ++i)
        {
            if (
                (i < n_visable) ||
                (i > vec.size() - n_visable - 1) ||
                (vec.size() <= 2*n_visable)
            )
            {
                os << ", " << static_cast<int>(vec[i]);
            }
            else if (i == n_visable)
            {
                os << ", ...";
            }
        }
    }
    os << "], dtype=" << type_name<T>() << ")";
    return os;
}


std::ostream& operator<<(std::ostream& os, const IdatParser& idata)
{
    std::map<int, int> offsets_ordered;
    for (const auto& pair : idata.offsets_) {
        offsets_ordered[static_cast<int>(pair.second)] = static_cast<int>(
            pair.first
        );
    }
    os << "file_size: " << idata.file_size_ << std::endl
       << "num_fields: " << idata.num_fields_ << std::endl;
    for (const auto& pair : offsets_ordered) {
        os << "\tkey: " << pair.second << "\toffset: " << pair.first << std::endl;
    }
    os << "illumina_ids: " << idata.illumina_ids_ << std::endl
       << "std_dev: " << idata.std_dev_ << std::endl
       << "probe_means: " << idata.probe_means_ << std::endl
       << "n_beads: " << idata.n_beads_ << std::endl
       << "mid_block: " << idata.mid_block_ << std::endl
       << "run_info: \n";
    for (size_t i = 0; i < idata.run_info_.size() / 5; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            os << idata.run_info_[i*5 + j] << ", ";
        }
        os << std::endl;
    }
    os << "red_green: " << idata.red_green_ << std::endl
       << "mostly_null: " << idata.mostly_null_ << std::endl
       << "barcode: " << idata.barcode_ << std::endl
       << "chip_type: " << idata.chip_type_ << std::endl
       << "mostly_a: " << idata.mostly_a_ << std::endl
       << "unknown_1: " << idata.unknown_1_ << std::endl
       << "unknown_2: " << idata.unknown_2_ << std::endl
       << "unknown_3: " << idata.unknown_3_ << std::endl
       << "unknown_4: " << idata.unknown_4_ << std::endl
       << "unknown_5: " << idata.unknown_5_ << std::endl
       << "unknown_6: " << idata.unknown_6_ << std::endl
       << "unknown_7: " << idata.unknown_7_ << std::endl;

    return os;
}

const std::string IdatParser::__str__() const {
    std::ostringstream os;
    os << "IdatParser(" << std::endl
       << "    file_size: " << file_size_ << std::endl
       << "    num_fields: " << num_fields_ << std::endl
       << "    illumina_ids: " << illumina_ids_ << std::endl
       << "    std_dev: " << std_dev_ << std::endl
       << "    probe_means: " << probe_means_ << std::endl
       << "    n_beads: " << n_beads_ << std::endl
       << "    mid_block: " << mid_block_ << std::endl
       << "    red_green: " << red_green_ << std::endl
       << "    mostly_null: " << mostly_null_ << std::endl
       << "    barcode: " << barcode_ << std::endl
       << "    chip_type: " << chip_type_ << std::endl
       << "    mostly_a: " << mostly_a_ << std::endl
       << "    unknown_1: " << unknown_1_ << std::endl
       << "    unknown_2: " << unknown_2_ << std::endl
       << "    unknown_3: " << unknown_3_ << std::endl
       << "    unknown_4: " << unknown_4_ << std::endl
       << "    unknown_5: " << unknown_5_ << std::endl
       << "    unknown_6: " << unknown_6_ << std::endl
       << "    unknown_7: " << unknown_7_ << std::endl
       << ")";
    return os.str();
}

const std::string IdatParser::__repr__() const {
    return __str__();
}
