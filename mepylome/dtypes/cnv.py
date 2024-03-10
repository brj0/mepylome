import numpy as np
import pandas as pd
import pyranges as pr
from cbseg import segment
from sklearn.linear_model import LinearRegression


class Annotation:
    def __init__(
        self,
        manifest,
        gap=None,
        genes=None,
        bin_size=50000,
        min_probes_per_bin=15,
        verbose=False,
    ):
        # TODO sort
        self.bin_size = bin_size
        self.min_probes_per_bin = min_probes_per_bin
        self.gap = gap
        self.genes = genes
        self.verbose = verbose
        self.chromsizes = pr.data.chromsizes()
        df = manifest.data_frame.copy()
        df = df[[x.startswith("cg") for x in df.Name.values]]
        df.rename(
            columns={"CHR": "Chromosome", "MAPINFO": "Start"}, inplace=True
        )
        df["End"] = df["Start"]
        df = df[
            df.Chromosome.isin(["X", "Y"] + [str(x) for x in range(1, 23)])
        ]
        df.Chromosome = "chr" + df.Chromosome
        self.manifest = pr.PyRanges(df)
        self.manifest = self.manifest.sort()
        self.probes = self.manifest.Name.values
        self.bins = self.make_bins()

    def make_bins(self):
        bins = pr.gf.tile_genome(self.chromsizes, int(5e4))
        bins = bins[bins.Chromosome != "chrM"]
        bins = bins.subtract(self.gap)
        bins = self.merge_bins(bins)
        return bins

    def merge_bins(self, bins):
        bins = bins.count_overlaps(
            self.manifest[["Name"]], overlap_col="N_probes"
        )
        bins = bins.apply(
            lambda df: self.merge_bins_in_chromosome(
                df, self.min_probes_per_bin, verbose=self.verbose
            ),
            as_pyranges=True,
        )
        return bins

    @staticmethod
    def merge_bins_in_chromosome(df, min_probes_per_bin, verbose=False):
        I_START = 0
        I_END = 1
        I_N_PROBES = 2
        INVALID = np.iinfo(np.int64).max

        matrix = df[["Start", "End", "N_probes"]].values.astype(np.int64)

        while np.any(matrix[:, I_N_PROBES] < min_probes_per_bin):
            i_min = np.argmin(matrix[:, I_N_PROBES])
            n_probes_left = INVALID
            n_probes_right = INVALID

            # Left
            if i_min > 0:
                delta_left = np.argmax(
                    matrix[i_min - 1 :: -1, I_N_PROBES] != INVALID
                )
                i_left = i_min - delta_left - 1
                if (
                    matrix[i_left, I_N_PROBES] != INVALID
                    and matrix[i_min, I_START] == matrix[i_left, I_END]
                ):
                    n_probes_left = matrix[i_left, I_N_PROBES]

            # Right
            if i_min < len(matrix) - 1:
                delta_right = np.argmax(
                    matrix[i_min + 1 :, I_N_PROBES] != INVALID
                )
                i_right = i_min + delta_right + 1
                if (
                    matrix[i_right, I_N_PROBES] != INVALID
                    and matrix[i_min, I_END] == matrix[i_right, I_START]
                ):
                    n_probes_right = matrix[i_right, I_N_PROBES]

            # Invalid
            if n_probes_left == INVALID and n_probes_right == INVALID:
                matrix[i_min, I_N_PROBES] = INVALID
                if verbose:
                    row = (
                        df.loc[i_min, ["Chromosome", "Start", "End"]]
                        .astype(str)
                        .tolist()
                    )
                    row_str = "-".join(row)
                    print(f"Could not merge {row_str}. Removed instead.")
                continue
            elif n_probes_right == INVALID or n_probes_left <= n_probes_right:
                i_merge = i_left
            else:
                i_merge = i_right
            matrix[i_merge, I_N_PROBES] += matrix[i_min, I_N_PROBES]
            matrix[i_merge, I_START] = min(
                matrix[i_merge, I_START], matrix[i_min, I_START]
            )
            matrix[i_merge, I_END] = max(
                matrix[i_merge, I_END], matrix[i_min, I_END]
            )
            matrix[i_min, I_N_PROBES] = INVALID
        df[["Start", "End", "N_probes"]] = matrix
        df = df[df.N_probes != INVALID]
        return df

    def __repr__(self):
        # TODO
        title = f"CNV():"
        lines = [
            title + "\n" + "*" * len(title),
            f"probes:\n{self.probes}",
            f"min_probes_per_bin:\n{self.min_probes_per_bin}",
            f"bin_size:\n{self.bin_size}",
            f"gap:\n{self.gap}",
            f"genes:\n{self.genes}",
            f"manifest:\n{self.manifest}",
            f"chromsizes:\n{self.chromsizes}",
            f"bins:\n{self.bins}",
        ]
        return "\n\n".join(lines)


class CNV:
    def __init__(self, sample, reference, annotation=None):
        self.sample = sample
        self.reference = reference
        self.annotation = annotation
        self.sample.intensity = self.get_itensity(self.sample)
        self.reference.intensity = self.get_itensity(self.reference)
        self.bins = annotation.bins

    def get_itensity(self, methyl_data):
        intensity = methyl_data.methylated + methyl_data.unmethylated
        prefix = "sample" if methyl_data == self.sample else "reference"
        if intensity.isna().any().any():
            print(f"{prefix}: Intensities that are NA, set to 1.")
            intensity.fillna(1, inplace=True)
        if (intensity < 1).any().any():
            print(f"{prefix}: Intensities smaller than 0 set to 1.")
            intensity[intensity < 1] = 1
        if intensity.mean(axis=0).min() < 5000:
            print(f"{prefix}: Intensities are abnormally low (< 5000).")
        if intensity.mean(axis=0).max() > 50000:
            print(f"{prefix}: Intensities are abnormally high (> 50000).")
        return intensity

    def fit(self):
        probes = self.annotation.probes
        smp_intensity = self.sample.intensity.loc[probes].values.ravel()
        idx = self.sample.intensity.loc[probes].index
        ref_intensity = self.reference.intensity.loc[
            probes,
        ].values
        correlation = np.array(
            [np.corrcoef(smp_intensity, z)[0, 1] for z in ref_intensity.T]
        )
        if any(correlation >= 0.99):
            LOGGER.warning(
                "Query sample also found in reference set. Excluded from fitting."
            )
        X = np.log2(ref_intensity)
        y = np.log2(smp_intensity)
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        y_pred[y_pred < 0] = 0
        self.coef = reg.coef_
        self._ratio = y - y_pred
        self.ratio = pd.DataFrame({"ratio": self._ratio}, idx)
        self.noise = np.sqrt(
            np.mean((self._ratio[1:] - self._ratio[:-1]) ** 2)
        )
        self.probes = self.annotation.manifest[["Name"]]
        self.probes.ratio = self._ratio

    def bin(self):
        self.bins.bins_index = np.arange(len(self.bins.df))
        overlap = self.bins.join(self.annotation.manifest[["Name"]])
        overlap.ratio = self.ratio.loc[overlap.Name].ratio
        result = overlap.df.groupby("bins_index", dropna=False)["ratio"].agg(
            ["median", "var"]
        )
        df = self.bins.df
        df["Median"] = np.nan
        df["Var"] = np.nan
        df.loc[result.index, ["Median", "Var"]] = result.values
        self.bins = pr.PyRanges(df)

    def genes(self):
        overlap = self.annotation.genes.join(
            self.annotation.manifest[["Name"]]
        )
        overlap.ratio = self.ratio.loc[overlap.Name].ratio
        result = overlap.df.groupby("Gene", dropna=False)["ratio"].agg(
            ["median", "var", "count"]
        )
        df = self.annotation.genes.df.set_index("Gene")
        df["Median"] = np.nan
        df["Var"] = np.nan
        df["N_probes"] = 0
        df.loc[result.index, ["Median", "Var", "N_probes"]] = result.values
        df["N_probes"] = df["N_probes"].astype(int)
        df = df.reset_index()
        self.genes = pr.PyRanges(df)
        self.genes = self.genes.sort()

    @staticmethod
    def get_segments(df):
        bin_values = df["Median"].values
        chrom = df["Chromosome"].iloc[0]
        seg = segment(bin_values, shuffles=1000, p=0.001)
        seg_df = pd.DataFrame(
            [
                [chrom, df.Start.iloc[l.start], df.End.iloc[l.end - 1]]
                for l in seg
            ],
            # [[chrom, l.start, l.end] for l in seg],
            columns=["Chromosome", "Start", "End"],
        )
        return seg_df

    def segments(self):
        segments = self.bins.apply(self.get_segments)
        overlap = segments.join(self.annotation.manifest[["Name"]])
        overlap.ratio = self.ratio.loc[overlap.Name].ratio
        result = (
            overlap.df.groupby(["Chromosome", "Start", "End"], dropna=False)[
                "ratio"
            ]
            .agg(["median", "mean", "var", "count"])
            .reset_index()
            .rename(
                columns={
                    "count": "N_probes",
                    "median": "Median",
                    "mean": "Mean",
                    "var": "Var",
                }
            )
        )
        self.segments = pr.PyRanges(result)
        self.segments = self.segments.sort()

    def __repr__(self):
        # TODO
        title = f"CNV():"
        lines = [
            title + "\n" + "*" * len(title),
            f"sample:\n{self.sample.probes}",
            f"reference:\n{self.reference.probes}",
            f"annotation: {self.annotation}",
            f"_ratio: {self._ratio}",
            f"bins:\n{self.bins}",
            f"genes:\n{self.genes}",
            f"segments:\n{self.segments}",
            f"coef:\n{self.coef}",
            f"noise:\n{self.noise}",
            f"ratio:\n{self.ratio}",
        ]
        return "\n\n".join(lines)
