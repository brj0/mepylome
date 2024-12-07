"""This script translates the commented rtd_tutorial.py file to rst format."""

from importlib.resources import files

PACKAGE_DIR = files("mepylome")


def parse_print_block(lines):
    result = []
    while lines[0] != '"""\n':
        line = lines.pop(0)
        result.append(f"    {line}")
    _ = lines.pop(0)
    result.append("\n")
    return result


def convert_py_to_rst(py_path, rst_path):
    """Converts a python file with comments to a rst file."""
    with open(py_path) as py_file:
        py_lines = py_file.readlines()
    rst_lines = []
    code_block = False
    while py_lines:
        line = py_lines.pop(0)
        if line == '"""\n':
            rst_lines.extend(parse_print_block(py_lines))
            continue
        # If line is a comment, convert it to reST paragraph
        if line.lstrip().startswith("# "):
            if code_block:
                rst_lines.append("\n")
                code_block = False
            comment = line[2:]
            rst_lines.append(comment)
        elif line.strip("#").strip() == "":
            rst_lines.append("\n")
        else:
            # If not a comment, include it as code
            if not code_block:
                rst_lines.append("\n.. code-block:: python\n\n")
                code_block = True
            if line.lstrip().startswith("##"):
                rst_lines.append("    >>> " + line.replace("##", "#", 1))
            else:
                rst_lines.append("    >>> " + line)
    with open(rst_path, "w") as rst_file:
        rst_file.writelines(rst_lines)


input_file = PACKAGE_DIR.parent / "examples" / "rtd_tutorial.py"
output_file = PACKAGE_DIR.parent / "docs" / "tutorial.rst"

convert_py_to_rst(input_file, output_file)
