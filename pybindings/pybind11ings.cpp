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

class PyIdatData : public IdatData {
public:
    // Constructor explicitly calling base class constructor
    // Without this line pybind11 generates a compiler error
    PyIdatData(const std::string& filepath) : IdatData(filepath) {}

    // PyIdatData(const fs::path& filepath) : IdatData(filepath.string()) {}

    // numpy compatible get functions for vectors
    py::array_t<int32_t> pyget_illumina_ids() const {
        return py::array_t<int32_t>(illumina_ids_.size(), illumina_ids_.data());
    }

    py::array_t<uint16_t> pyget_std_dev() const {
        return py::array_t<uint16_t>(std_dev_.size(), std_dev_.data());
    }

    py::array_t<int16_t> pyget_probe_means() const {
        return py::array_t<int16_t>(probe_means_.size(), probe_means_.data());
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
    py::class_<PyIdatData>(m, "IdatData")

        .def(
            py::init< const std::string& >(),
            py::arg("filepath")
        )
        // .def(
            // py::init<const fs::path &>(),
            // py::arg("filepath")
        // )
        .def("__str__", &PyIdatData::__str__)
        .def("__repr__", &PyIdatData::__repr__)

        // numpy arrays
        .def_property_readonly("illumina_ids", &PyIdatData::pyget_illumina_ids)
        .def_property_readonly("std_dev", &PyIdatData::pyget_std_dev)
        .def_property_readonly("probe_means", &PyIdatData::pyget_probe_means)
        .def_property_readonly("n_beads", &PyIdatData::pyget_n_beads)
        .def_property_readonly("mid_block", &PyIdatData::pyget_mid_block)

        // list
        .def_property_readonly("run_info", &PyIdatData::get_run_info)
        .def_property_readonly("offsets", &PyIdatData::get_offsets)

        // int
        .def_property_readonly("file_size", &PyIdatData::get_file_size)
        .def_property_readonly("num_fields", &PyIdatData::get_num_fields)
        .def_property_readonly("n_snps_read", &PyIdatData::get_n_snps_read)
        .def_property_readonly("red_green", &PyIdatData::get_red_green)

        // strings
        .def_property_readonly("mostly_null", &PyIdatData::get_mostly_null)
        .def_property_readonly("barcode", &PyIdatData::get_barcode)
        .def_property_readonly("chip_type", &PyIdatData::get_chip_type)
        .def_property_readonly("mostly_a", &PyIdatData::get_mostly_a)
        .def_property_readonly("unknown_1", &PyIdatData::get_unknown_1)
        .def_property_readonly("unknown_2", &PyIdatData::get_unknown_2)
        .def_property_readonly("unknown_3", &PyIdatData::get_unknown_3)
        .def_property_readonly("unknown_4", &PyIdatData::get_unknown_4)
        .def_property_readonly("unknown_5", &PyIdatData::get_unknown_5)
        .def_property_readonly("unknown_6", &PyIdatData::get_unknown_6)
        .def_property_readonly("unknown_7", &PyIdatData::get_unknown_7)
    ;
}
