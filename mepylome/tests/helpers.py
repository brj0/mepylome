"""Helper functions for unittests."""

import gzip
import uuid

from mepylome.tests.write_idat import IdatWriter
from mepylome.utils.files import (
    ensure_directory_exists,
)
from mepylome.utils.varia import MEPYLOME_TMP_DIR

# Create a temporary test directory
TEST_DIR = MEPYLOME_TMP_DIR / "tests"
ensure_directory_exists(TEST_DIR)


def _write_binary(path, value):
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as gz_file:
            gz_file.write(value)
    else:
        with open(path, "wb") as file:
            file.write(value)


class TempIdatFile:
    """Creates a temporary IDAT file."""

    def __init__(self, data, gzipped=False):
        suffix = "idat.gz" if gzipped else ".idat"
        basename = str(uuid.uuid4())
        self.path = TEST_DIR / (basename + suffix)

        idat_writer = IdatWriter(data=data)
        self.data = idat_writer.data

        # Write the IDAT data to the file
        _write_binary(self.path, idat_writer.buffer.getvalue())

    def __del__(self):
        self.path.unlink()


class TempIdatFilePair:
    """Creates a temporary IDAT file pair."""

    def __init__(self, data_grn, data_red, gzipped=False):
        basename = str(uuid.uuid4())
        suffix = "idat.gz" if gzipped else ".idat"
        self.basepath = TEST_DIR / basename
        self.path_grn = TEST_DIR / (basename + "_Grn" + suffix)
        self.path_red = TEST_DIR / (basename + "_Red" + suffix)

        idat_writer_grn = IdatWriter(data=data_grn)
        self.data_grn = idat_writer_grn.data

        idat_writer_red = IdatWriter(data=data_red)
        self.data_red = idat_writer_red.data

        # Write the IDAT data to the file
        _write_binary(self.path_grn, idat_writer_grn.buffer.getvalue())
        _write_binary(self.path_red, idat_writer_red.buffer.getvalue())

    def __del__(self):
        self.path_grn.unlink()
        self.path_red.unlink()


TEST_MANIFEST_CSV = b"""Illumina, Inc.,,,,,,,,,
[Heading],,,,,,,,,,
Descriptor File Name,BS0010894-AQP_content.bpm,,,,,,,,,
Assay Format,Infinium 2,,,,,,,,,
Date Manufactured,6/11/2008,,,,,,,,,
Loci Count ,485553,,,,,,,,,
[Assay],,,,,,,,,,
IlmnID,Name,AddressA_ID,AlleleA_ProbeSeq,AddressB_ID,AlleleB_ProbeSeq,Infinium_Design_Type,Color_Channel,CHR,MAPINFO,Strand
cg00035864,cg00035864,31729416,AAAACACTAACAATCTTATCCACATAAACCCTTAAATTTATCTCAAATTC,,,II,,Y,8553009,F
cg21776599,cg21776599,12751494,ATCTAAATAAAACCTTTCAAAATCCTCCTAAAATCAACATCCRAATAACC,,,II,,1,1830693,F
cg10651537,cg10651537,22700311,ACTTTTTAAATACCAAAACCCAAAAACTACCACCACTCTTAAACTAAAAC,,,II,,2,177134737,F
cg03653856,cg03653856,64649502,ACTAAATTAAAAATTTCTAAAATAACACAAACCCCTCCACTACACTCACA,71783336,ACTAAATTAAAAATTTCTAAAATAACGCGAACCCCTCCGCTACACTCACG,I,Grn,3,197500859,R
cg03738331,cg03738331,29800410,CTACRAAAACAAAAACRAACTAATTCCCTAACCAACCATTAACAAAATCC,,,II,,4,3076305,R
cg07525751,cg07525751,35667488,TTCCAAAAATTATTAAAACACTACAATCATATCATATATAAATAAAACCA,55682328,TTCCGAAAATTATTAAAACGCTACGATCATATCATATATAAATAAAACCG,I,Red,5,1416958,F
cg04704193,cg04704193,14708316,TCCTAAAATATCATCAATAAAACCCAACAAAAAATAAATAATAAAAATCA,52752332,TCCTAAAATATCATCAATAAAACCCAACGAAAAATAAATAATAAAAATCG,I,Red,6,26272200,F
cg17193551,cg17193551,66721465,CAATTTAACTCCAAAACCCRTACTTTATTTATACTAATACTACTCAAAAC,,,II,,8,130925897,F
cg14368972,cg14368972,69739459,TAACAACAAAACAAAACTAACTTCATAACAAATCAATTTCCCTCTCAACA,22775438,TAACAACGAAACAAAACTAACTTCATAACAAATCAATTTCCCTCTCAACG,I,Red,9,2717146,F
cg07175255,cg07175255,72782355,TCAAAACACTAAATAATTAAAAATATCTCTTCAACAAAAAAACAATACCA,11622354,TCAAAACACTAAATAATTAAAAATATCTCTTCGACGAAAAAACGATACCG,I,Red,10,131218009,F
cg04398180,cg04398180,62793332,ATCTTATAAAACRCTAAACACRACTACTTCCTAATTCAAAATCTACTCTC,,,II,,13,114103452,R
cg25598086,cg25598086,48773376,CTCCTAAAATTTTATAATAAATACTTCACTACRTCTCAATAATTATAAAC,,,II,,14,65006720,F
cg08354699,cg08354699,16765420,CTCCATAACCACAAAAAAAACAACTAAAAAAACTACAACTCCAAAACCCA,32725388,CTCCATAACCACGAAAAAAACAACTAAAAAAACTACGACTCCAAAACCCG,I,Red,15,63796895,
cg09913882,cg09913882,18687439,RAATAAACACCRTTAATACCCAACTTAACCAAAACCTCRTCCTTAAAATC,,,II,,17,73612981,F
cg01017773,cg01017773,30736309,ATATCTAAAACAATAAAAACRCTCRTATTTAAAACCCATTTTAACAACAC,,,II,,19,57988497,F
cg27509521,cg27509521,12654498,ACRTAACAACCAATAAACRCATAAATTAACTAAATACRACCAATAAAAAC,,,II,,19,17420279,R
rs10796216,rs10796216,14622465,TAACTAAAAAACAACAATACTAACTCTACACTAAATACCCACTAACCCTC,41635319,TAACTAAAAAACAACAATACTAACTCTACACTAAATACCCACTAACCCTT,I,Red,,,
rs715359,rs715359,18796328,TTATTAAACTCTCACCACTAACTTTCTACTTCTCTCAAAATCAAAACCTC,48710462,TTATTAAACTCTCACCACTAACTTTCTACTTCTCTCAAAATCAAAACCTT,I,Grn,,,
[Controls],,,,,,,,,,
34648333,STAINING,Blue,Biotin (Bkg),,,,,,,
31698466,EXTENSION,Blue,Extension (G),,,,,,,
26772442,HYBRIDIZATION,Blue,Hyb (Medium),,,,,,,
21771417,HYBRIDIZATION,Black,Hyb (Low),,,,,,,
13643320,TARGET REMOVAL,Green,Target Removal 1,,,,,,,
22795447,BISULFITE CONVERSION I,LimeGreen,BS Conversion I-C2,,,,,,,
65797428,SPECIFICITY I,Gold,GT Mismatch 5 (MM),,,,,,,
17661470,SPECIFICITY II,Red,Specificity 2,,,,,,,
70645401,NON-POLYMORPHIC,Blue,NP (G),,,,,,,
13792480,NEGATIVE,Red,Negative 1,AVG,,,,,,
74737439,NEGATIVE,Maroon,Negative 5,,,,,,,
40758350,NEGATIVE,Tan,Negative 9,,,,,,,
74688409,NEGATIVE,Teal,Negative 16,,,,,,,
30689326,NEGATIVE,BlueViolet,Negative 532,,,,,,,
18733503,NORM_G,Blue,Norm_G1,,,,,,,
70603486,NORM_A,Red,Norm_A2,,,,,,,
63775337,NORM_G,Blue,Norm_G2,,,,,,,
70622474,NORM_A,Red,Norm_A3,,,,,,,
58676498,NORM_G,Blue,Norm_G3,,,,,,,
17630478,NORM_A,Red,Norm_A4,,,,,,,
35689446,NORM_G,Blue,Norm_G4,,,,,,,
15639322,NORM_A,Red,Norm_A5,,,,,,,
61760309,NORM_G,Blue,Norm_G5,,,,,,,
66624473,NORM_A,Red,Norm_A6,,,,,,,"""


class TempManifest:
    """Creates a temporary manifest file."""

    def __init__(self):
        basename = str(uuid.uuid4())
        self.path = TEST_DIR / ("tmp_manifest_" + basename + ".csv")
        _write_binary(self.path, TEST_MANIFEST_CSV)

    def __del__(self):
        self.path.unlink()
