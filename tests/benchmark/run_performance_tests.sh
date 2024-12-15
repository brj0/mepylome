#!/bin/bash

# Define the preprocessing methods
preps=("illumina" "noob" "raw" "swan")

# Define the base directory
BASE_DIR="$HOME/mepylome/tests"

# Get the list of subdirectories
subdirs=($(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d))

print_separator() {
    echo ""
    echo ""
    echo "***************************************************************"
}

# Trap Ctrl+C and exit gracefully
trap ctrl_c INT

ctrl_c() {
    echo "Ctrl+C detected. Exiting..."
    exit 1
}

# Loop over each preprocessing method
for prep in "${preps[@]}"; do
    print_separator
    echo "mepylome: python test_idat_extraction.py $prep"
    /usr/bin/time -v python test_idat_extraction.py "$prep"

    print_separator
    echo "minfi: Rscript test_idat_extraction.R $prep"
    /usr/bin/time -v Rscript test_idat_extraction.R "$prep"

    # Loop over each subdirectory
    for subdir in "${subdirs[@]}"; do

        # Extract the subdirectory name
        subdir_name=$(basename "$subdir")

        print_separator
        echo "mepylome: python test_cnv.py $prep $subdir_name"
        /usr/bin/time -v python test_cnv.py "$prep" "$subdir_name"

        print_separator
        echo "conumee2.0: Rscrip test_cnv.R $prep $subdir_name"
        /usr/bin/time -v Rscript test_cnv.R "$prep" "$subdir_name"
    done
done
