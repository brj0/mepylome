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


class IdatParser
{
public:
    size_t file_size_;
    uint32_t num_fields_;
    std::unordered_map<IdatSectionCode, std::streamoff> offsets_;
    int32_t n_snps_read_;
    std::vector<int32_t> illumina_ids_;
    std::vector<uint16_t> std_dev_;
    std::vector<uint16_t> probe_means_;
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
    bool intensity_only;
    bool array_type_only;


    IdatParser(
        const std::string& filepath,
        bool intensity_only = false,
        bool array_type_only = false
    );

    size_t get_file_size() const { return file_size_; }
    uint32_t get_num_fields() const { return num_fields_; }
    const std::unordered_map<
        IdatSectionCode, std::streamoff
    >& get_offsets() const { return offsets_; }
    int32_t get_n_snps_read() const { return n_snps_read_; }
    const std::vector<int32_t>& get_illumina_ids() const { return illumina_ids_; }
    const std::vector<uint16_t>& get_std_dev() const { return std_dev_; }
    const std::vector<uint16_t>& get_probe_means() const { return probe_means_; }
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

    const std::string __str__() const;
    const std::string __repr__() const;

    friend std::ostream& operator<<(std::ostream& os, const IdatParser& idata);

};
