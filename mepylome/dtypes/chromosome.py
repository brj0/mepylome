"""Defines an Enum representing chromosomes with utility methods.

Usage:
    chr_2 = Chromosome.CHR2
    chr_x = Chromosome.CHRX

    is_valid = Chromosome.is_valid_chromosome(chr_x)
    df = pd.DataFrame({'Chromosome_0': ["13", "chr1", "chr2", "chrY", "chrx"]})
    df['Chromosome_1'] = Chromosome.pd_from_string(df['Chromosome_0'])
    df['Chromosome_2'] = Chromosome.pd_to_string(df['Chromosome_1'])
"""

from enum import IntEnum, unique


@unique
class Chromosome(IntEnum):
    """Enum representing chromosomes and providing utility methods.

    Chromosomes are represented as integers:
    - 0-22: Chromosomes 1-22
    - 23: Chromosome X
    - 24: Chromosome Y
    - 25: Mitochondrial Chromosome (MT)
    - -1: Invalid Chromosome

    Methods:
        - is_valid_chromosome(chrom): Checks if the given chromosome is valid.
        - pd_from_string(col): Converts chromosome strings to Chromosome enum
            values.
        - pd_to_string(col): Converts Chromosome enum values to chromosome
            strings.
    """

    CHR0 = 0
    CHR1 = 1
    CHR2 = 2
    CHR3 = 3
    CHR4 = 4
    CHR5 = 5
    CHR6 = 6
    CHR7 = 7
    CHR8 = 8
    CHR9 = 9
    CHR10 = 10
    CHR11 = 11
    CHR12 = 12
    CHR13 = 13
    CHR14 = 14
    CHR15 = 15
    CHR16 = 16
    CHR17 = 17
    CHR18 = 18
    CHR19 = 19
    CHR20 = 20
    CHR21 = 21
    CHR22 = 22
    CHRX = 23
    CHRY = 24
    CHRM = 25
    INVALID = -1

    @staticmethod
    def is_valid_chromosome(chrom):
        """Checks if the given chromosome is valid."""
        return (chrom >= Chromosome.CHR1) & (chrom <= Chromosome.CHRY)

    @staticmethod
    def pd_from_string(col):
        """Converts chromosome strings to Chromosome enum values."""
        chrom_map = {
            **{str(i): Chromosome(i) for i in range(23)},
            **{"chr" + str(i): Chromosome(i) for i in range(23)},
            "X": Chromosome.CHRX,
            "Y": Chromosome.CHRY,
            "M": Chromosome.CHRM,
            "x": Chromosome.CHRX,
            "y": Chromosome.CHRY,
            "m": Chromosome.CHRM,
            "chrX": Chromosome.CHRX,
            "chrY": Chromosome.CHRY,
            "chrM": Chromosome.CHRM,
            "chrx": Chromosome.CHRX,
            "chry": Chromosome.CHRY,
            "chrm": Chromosome.CHRM,
        }
        return col.map(chrom_map).fillna(Chromosome.INVALID).astype(int)

    @staticmethod
    def pd_to_string(col):
        """Converts Chromosome enum values to chromosome strings."""
        chrom_map = {
            **{Chromosome(i): "chr" + str(i) for i in range(23)},
            Chromosome.CHRX: "chrX",
            Chromosome.CHRY: "chrY",
            Chromosome.CHRM: "chrM",
        }
        return col.map(chrom_map).fillna("NaN")
