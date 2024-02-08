

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "This code is not compatible with big-endian systems."
#endif


#include "parser.h"


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

IdatData::IdatData(const std::string& filepath)
    : file_size_(static_cast<size_t>(filesize(filepath)))
{
    // Start parsing IDAT file

    std::ifstream idat_file(filepath, std::ios::binary);

    if (!idat_file.is_open())
    {
        std::cerr << "Error opening file: " << filepath << std::endl;
    }


    // Parse IDAT header

    std::string file_type = read_char(idat_file, std::strlen(DEFAULT_IDAT_FILE_ID));

    if (file_type != DEFAULT_IDAT_FILE_ID)
    {
        std::cerr << "Not an IDAT file. Unsupported file type."
                    << std::endl;
    }

    uint64_t idat_version = read_long(idat_file);

    if (idat_version != DEFAULT_IDAT_VERSION)
    {
        std::cerr << "Not a version 3 IDAT file. Unsupported IDAT version."
                    << std::endl;
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

    idat_file.seekg(offsets_[IdatSectionCode::ILLUMINA_ID]);
    illumina_ids_ = read_array<int32_t>(idat_file, n_snps_read_);

    idat_file.seekg(offsets_[IdatSectionCode::STD_DEV]);
    std_dev_= read_short(idat_file, n_snps_read_);

    idat_file.seekg(offsets_[IdatSectionCode::MEAN]);
    probe_means_= read_array<int16_t>(idat_file, n_snps_read_);

    idat_file.seekg(offsets_[IdatSectionCode::NUM_BEADS]);
    n_beads_= read_array<uint8_t>(idat_file, n_snps_read_);

    idat_file.seekg(offsets_[IdatSectionCode::MID_BLOCK]);
    int32_t n_mid_block = read_int(idat_file);
    mid_block_ = read_int(idat_file, n_mid_block);

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

bool is_big_endian()
{
    uint32_t value = 1;
    return (*reinterpret_cast<uint8_t*>(&value) == 0);
}

std::ostream& operator<<(std::ostream& os, const IdatData& idata) {
    std::map<int, int> offsets_ordered;
    for (const auto& pair : idata.offsets_) {
        offsets_ordered[static_cast<int>(pair.second)] = static_cast<int>(
            pair.first
        );
    }
    os << "is_big_endian: " << is_big_endian() << "\n"
       << "file_size: " << idata.file_size_ << std::endl
       << "Number of Fields: " << idata.num_fields_ << std::endl;
    for (const auto& pair : offsets_ordered) {
        os << "\tkey: " << pair.second << "\toffset: " << pair.first << std::endl;
    }
    os << "illumina_ids: " << idata.illumina_ids_ << std::endl;
    os << "std_dev: " << idata.std_dev_ << std::endl;
    os << "probe_means: " << idata.probe_means_ << std::endl;
    os << "n_beads: " << idata.n_beads_ << std::endl;
    os << "mid_block: " << idata.mid_block_ << std::endl;
    std::cout << "run_info: \n";
    for (size_t i = 0; i < idata.run_info_.size() / 5; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            std::cout << idata.run_info_[i*5 + j] << ", ";
        }
        std::cout << std::endl;
    }
    os << "red_green: " << idata.red_green_ << std::endl;
    os << "mostly_null: " << idata.mostly_null_ << std::endl;
    os << "barcode: " << idata.barcode_ << std::endl;
    os << "chip_type: " << idata.chip_type_ << std::endl;
    os << "mostly_a: " << idata.mostly_a_ << std::endl;
    os << "unknown_1: " << idata.unknown_1_ << std::endl;
    os << "unknown_2: " << idata.unknown_2_ << std::endl;
    os << "unknown_3: " << idata.unknown_3_ << std::endl;
    os << "unknown_4: " << idata.unknown_4_ << std::endl;
    os << "unknown_5: " << idata.unknown_5_ << std::endl;
    os << "unknown_6: " << idata.unknown_6_ << std::endl;
    os << "unknown_7: " << idata.unknown_7_ << std::endl;

    return os;
}

int main()
{
    std::string idat_path = (std::string)getenv("HOME") +
        "/MEGA/work/programming/pyllumina/101130760092_R05C02_Grn.idat";

    Timer timer;
    IdatData idat_data(idat_path);
    timer.stop("Time for parsing");

    std::cout << idat_data << "\n";
}
