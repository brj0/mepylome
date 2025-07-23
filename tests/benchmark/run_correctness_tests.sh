#!/bin/bash

# Define the preprocessing methods
preps=("illumina" "noob" "raw" "swan")

# Define the base directory
BASE_DIR="$HOME/mepylome/tests/"

# Get the list of subdirectories
subdirs=($(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d))

print_separator() {
    echo ""
    echo ""
    echo "***************************************************************"
}

# Loop over each preprocessing method
for prep in "${preps[@]}"; do
    print_separator
    echo "mepylome: python test_idat_extraction.py $prep --save"
    python test_idat_extraction.py "$prep" --save

    print_separator
    echo "minfi: Rscript test_idat_extraction.R $prep --save"
    Rscript test_idat_extraction.R "$prep" --save

done

print_separator
print_separator
echo "Compare saved output."
python correctness.py
