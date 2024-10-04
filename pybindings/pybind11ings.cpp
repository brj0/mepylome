/*
 * @file pybind11ings.cpp
 *
 * @brief Contaings the python bindings.
 */


// #include <filesystem>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/stl/filesystem.h>
#include <string>

#include "../src/idat_parser.h"


namespace py = pybind11;
// namespace fs = std::filesystem;

class PyIdatParser : public IdatParser {
public:
    // Constructor explicitly calling base class constructor
    // Without this line pybind11 generates a compiler error
    PyIdatParser(
        const std::string& filepath,
        bool intensity_only,
        bool array_type_only
    ) : IdatParser(filepath, intensity_only, array_type_only) {}

    // PyIdatParser(const fs::path& filepath) : IdatParser(filepath.string()) {}

    // numpy compatible get functions for vectors
    py::array_t<int32_t> pyget_illumina_ids() const {
        return py::array_t<int32_t>(illumina_ids_.size(), illumina_ids_.data());
    }

    py::array_t<uint16_t> pyget_std_dev() const {
        return py::array_t<uint16_t>(std_dev_.size(), std_dev_.data());
    }

    py::array_t<uint16_t> pyget_probe_means() const {
        return py::array_t<uint16_t>(probe_means_.size(), probe_means_.data());
    }

    py::array_t<uint8_t> pyget_n_beads() const {
        return py::array_t<uint8_t>(n_beads_.size(), n_beads_.data());
    }

    py::array_t<int32_t> pyget_mid_block() const {
        return py::array_t<int32_t>(mid_block_.size(), mid_block_.data());
    }

};

PYBIND11_MODULE(_mepylome, m)
{
    m.doc() = "Parses idat files";
    py::class_<PyIdatParser>(m, "IdatParser")

        .def(
            py::init<const std::string&, bool, bool>(),
            py::arg("filepath"),
            py::arg("intensity_only") = false,
            py::arg("array_type_only") = false
        )
        // .def(
            // py::init<const fs::path &>(),
            // py::arg("filepath")
        // )
        .def("__str__", &PyIdatParser::__str__)
        .def("__repr__", &PyIdatParser::__repr__)

        // numpy arrays
        .def_property_readonly("illumina_ids", &PyIdatParser::pyget_illumina_ids)
        .def_property_readonly("std_dev", &PyIdatParser::pyget_std_dev)
        .def_property_readonly("probe_means", &PyIdatParser::pyget_probe_means)
        .def_property_readonly("n_beads", &PyIdatParser::pyget_n_beads)
        .def_property_readonly("mid_block", &PyIdatParser::pyget_mid_block)

        // list
        .def_property_readonly("run_info", &PyIdatParser::get_run_info)
        .def_property_readonly("offsets", &PyIdatParser::get_offsets)

        // int
        .def_property_readonly("file_size", &PyIdatParser::get_file_size)
        .def_property_readonly("num_fields", &PyIdatParser::get_num_fields)
        .def_property_readonly("n_snps_read", &PyIdatParser::get_n_snps_read)
        .def_property_readonly("red_green", &PyIdatParser::get_red_green)

        // strings
        .def_property_readonly("mostly_null", &PyIdatParser::get_mostly_null)
        .def_property_readonly("barcode", &PyIdatParser::get_barcode)
        .def_property_readonly("chip_type", &PyIdatParser::get_chip_type)
        .def_property_readonly("mostly_a", &PyIdatParser::get_mostly_a)
        .def_property_readonly("unknown_1", &PyIdatParser::get_unknown_1)
        .def_property_readonly("unknown_2", &PyIdatParser::get_unknown_2)
        .def_property_readonly("unknown_3", &PyIdatParser::get_unknown_3)
        .def_property_readonly("unknown_4", &PyIdatParser::get_unknown_4)
        .def_property_readonly("unknown_5", &PyIdatParser::get_unknown_5)
        .def_property_readonly("unknown_6", &PyIdatParser::get_unknown_6)
        .def_property_readonly("unknown_7", &PyIdatParser::get_unknown_7)
    ;
}
