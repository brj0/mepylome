from pathlib import Path

import pkg_resources

PACKAGE_DIR = Path(pkg_resources.resource_filename("mepylome", ""))

def convert_py_to_rst(py_path, rst_path):
    with open(py_path) as py_file:
        py_lines = py_file.readlines()
    rst_lines = []
    code_block = False
    for line in py_lines:
        # If line is a comment, convert it to reST paragraph
        if line.lstrip().startswith('# '):
            if code_block:
                rst_lines.append("\n")
                code_block = False
            comment = line[2:]
            rst_lines.append(comment)
        elif line.strip("#").strip() == "":
            if code_block:
                rst_lines.append('\n')
            else:
                rst_lines.append('\n')
        else:
            # If not a comment, include it as code
            if not code_block:
                rst_lines.append("\n.. code-block:: python\n\n")
                code_block = True
            if line.lstrip().startswith('##'):
                rst_lines.append("    >>> " + line[1:])
            else:
                rst_lines.append("    >>> " + line)
    with open(rst_path, 'w') as rst_file:
        rst_file.writelines(rst_lines)

input_file = PACKAGE_DIR.parent / "tests" / "tutorial_basic.py"
output_file = PACKAGE_DIR.parent / "docs" / "tutorial.rst"

convert_py_to_rst(input_file, output_file)