
    Read idat_version (8 bytes, little-endian)
    char idat_version_buffer[9];
    idat_file.read(idat_version_buffer, 8);
    idat_version_buffer[8] = '\0';
    int64_t idat_version = *reinterpret_cast<int64_t*>(idat_version_buffer);

inline uint64_t read_long(std::ifstream& infile)
{
    char buffer[8];
    infile.read(buffer, 8);
    uint64_t result = *reinterpret_cast<uint64_t*>(buffer);
    delete[] buffer;
    return result;
}

    // Read num_fields (4 bytes, little-endian)
    char num_fields_buffer[5];
    idat_file.read(num_fields_buffer, 4);
    num_fields_buffer[4] = '\0';
    int32_t num_fields = *reinterpret_cast<int32_t*>(num_fields_buffer);



inline uint16_t read_short(std::ifstream& infile) {
    char buffer[2];
    infile.read(buffer, 2);
    return static_cast<uint16_t>(
        (static_cast<uint8_t>(buffer[1]) << 8) |
        static_cast<uint8_t>(buffer[0])
    );
}

// Read 4 bytes in little endian format to int.
inline int32_t read_int(std::ifstream& infile) {
    char buffer[4];
    infile.read(buffer, 4);
    return static_cast<int32_t>(
        (static_cast<uint8_t>(buffer[3]) << 24) |
        (static_cast<uint8_t>(buffer[2]) << 16) |
        (static_cast<uint8_t>(buffer[1]) << 8) |
        static_cast<uint8_t>(buffer[0])
    );
}


    std::cout << " buffer(0,1)=(" << static_cast<int16_t>(buffer[0])
         << "," << static_cast<int16_t>(buffer[1]) << ")";
    std::bitset<8> b0(buffer[0]);
    std::bitset<8> b1(buffer[1]);
    std::cout << " buffer(0,1)=(" << b0
         << "," << b1 << ")";
