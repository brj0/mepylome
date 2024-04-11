self = Manifest("450k")
epic = Manifest("epic")
epicv2 = Manifest("epicv2")



from numba import njit
@njit
def numba_merge_bins(matrix, min_probes_per_bin, verbose=False):
    I_START = 0
    I_END = 1
    I_N_PROBES = 2
    INVALID = np.iinfo(np.int64).max
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
            delta_right = np.argmax(matrix[i_min + 1 :, I_N_PROBES] != INVALID)
            i_right = i_min + delta_right + 1
            if (
                matrix[i_right, I_N_PROBES] != INVALID
                and matrix[i_min, I_END] == matrix[i_right, I_START]
            ):
                n_probes_right = matrix[i_right, I_N_PROBES]
        # Invalid
        if n_probes_left == INVALID and n_probes_right == INVALID:
            matrix[i_min, I_N_PROBES] = INVALID
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
    return matrix

# z = numba_merge_bins(
#     df[["Start", "End", "n_probes"]].values.astype(np.int64),
#     min_probes_per_bin,
#     verbose=True,
# )




swan = np.full((len(self.probes), len(self.methyl_index)), np.nan)
for i in range(len(self.probes)):
    for probe_type in [ProbeType.ONE, ProbeType.TWO]:
        curr_intensity = intensity[i, all_indices[probe_type]]
        x = rankdata(curr_intensity) / len(curr_intensity)
        xp = np.sort(x[random_indices[probe_type]])
        fp = sorted_subset_intensity[i,:]
        # xp = intensity[i,random_indices[probe_type]]
        # x = normed_rank[i,:]
        intensity_min = np.min(curr_intensity[random_indices[probe_type]])
        intensity_max = np.max(curr_intensity[random_indices[probe_type]])
        i_max = np.where(x > np.max(xp))
        i_min = np.where(x < np.min(xp))
        delta_max = curr_intensity[i_max] - intensity_max
        delta_min = curr_intensity[i_min] - intensity_min
        interp = np.interp(x=x, xp=xp, fp=fp)
        interp[i_max] = np.max(fp) + delta_max
        interp[i_min] = np.min(fp) + delta_min
        interp = np.where(interp <= 0, bg_intensity[i], interp)
        swan[i, all_indices[probe_type]] = interp



################### NOOB

from scipy.stats import norm


def huber(y, k=1.5, tol=1.0e-6):
    y = y[~np.isnan(y)]
    n = len(y)
    mu = np.median(y)
    s = np.median(np.abs(y - mu)) * 1.4826
    if s == 0:
        raise ValueError("Cannot estimate scale: MAD is zero for this sample")
    while True:
        yy = np.clip(y, mu - k * s, mu + k * s)
        mu1 = np.sum(yy) / n
        if np.abs(mu - mu1) < tol * s:
            break
        mu = mu1
    return mu, s


def normexp_signal(par, x):
    mu = par[0]
    sigma = np.exp(par[1])
    sigma2 = sigma * sigma
    alpha = np.exp(par[2])
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mu_sf = x - mu - sigma2 / alpha
    log_dnorm = norm.logpdf(0, loc=mu_sf, scale=sigma)
    log_pnorm = norm.logsf(0, loc=mu_sf, scale=sigma)
    signal = mu_sf + sigma2 * np.exp(log_dnorm - log_pnorm)
    o = ~np.isnan(signal)
    if np.any(signal[o] < 0):
        print(
            "Limit of numerical accuracy reached with very low intensity or "
            "very high background:\nsetting adjusted intensities to small "
            "value"
        )
        signal[o] = np.maximum(signal[o], 1e-6)
    return signal


def normexp_get_xs(xf, controls=None, offset=50, param=None):
    xf_idx = xf.index
    xf_cols = xf.columns
    result = np.empty(xf.shape)
    n_probes = xf.shape[1]
    if param is None:
        if controls is None:
            ValueError("'controls' or 'param' must be given")
        alpha = np.empty(n_probes)
        mu = np.empty(n_probes)
        sigma = np.empty(n_probes)
        for i in range(n_probes):
            mu[i], sigma[i] = huber(controls[:, i])
            alpha[i] = max(huber(xf.values[:, i])[0] - mu[i], 10)
        param = np.column_stack((mu, np.log(sigma), np.log(alpha)))
    for i in range(n_probes):
        result[:, i] = normexp_signal(param[i], xf.values[:, i])
    return {
        "xs": (result + offset).T,
        "param": param,
    }


offset = 15
dye_corr = True
verbose = False
dye_method = "single"

raw = RawData([ref0, ref1])
self = MethylData(raw, prep="raw")

timer.start()

grn = raw.grn
red = raw.red

timer.stop("0")

i_grn = self.manifest.probe_info(ProbeType.ONE, Channel.GRN)
i_red = self.manifest.probe_info(ProbeType.ONE, Channel.RED)

timer.stop("1")

grn_oob = pd.concat(
    [grn.loc[i_red.AddressA_ID], grn.loc[i_red.AddressB_ID]], axis=0
)
red_oob = pd.concat(
    [red.loc[i_grn.AddressA_ID], red.loc[i_grn.AddressB_ID]], axis=0
)

timer.stop("2.0")

control_probes = self.manifest.control_data_frame

timer.stop("2.1")

control_probes = control_probes[
    control_probes.Address_ID.isin(red.index)
].reset_index(drop=True)

timer.stop("3")

self.methylated[self.methylated <= 0] = 1
self.unmethylated[self.unmethylated <= 0] = 1

timer.stop("4.0")

manifest_df = self.manifest.data_frame.iloc[self.methyl_index]

timer.stop("4.1")

probe_type = manifest_df.Probe_Type
color = manifest_df.Color_Channel

timer.stop("5")

ext_probe_type = np_ext_probe_type(probe_type, color)

timer.stop("6")

i_grn_idx = manifest_df.index[ext_probe_type == ExtProbeType.ONE_GRN]
i_red_idx = manifest_df.index[ext_probe_type == ExtProbeType.ONE_RED]
ii_idx = manifest_df.index[ext_probe_type == ExtProbeType.TWO]

timer.stop("7.0")

grn_m = self.methylated.iloc[i_grn_idx]
grn_u = self.unmethylated.iloc[i_grn_idx]
grn_2 = self.methylated.iloc[ii_idx]

timer.stop("7.1")

xf_grn = pd.concat([grn_m, grn_u, grn_2], axis=0)

timer.stop("7.2")

xs_grn = normexp_get_xs(xf_grn, controls=grn_oob.values, offset=offset)

timer.stop("8")

cumsum = np.cumsum([0, len(grn_m), len(grn_u), len(grn_2)])
range_grn_m = range(cumsum[0], cumsum[1])
range_grn_u = range(cumsum[1], cumsum[2])
range_grn_2 = range(cumsum[2], cumsum[3])

timer.stop("9.0")

red_m = self.methylated.iloc[i_red_idx]
red_u = self.unmethylated.iloc[i_red_idx]
red_2 = self.unmethylated.iloc[ii_idx]

timer.stop("9.1")

xf_red = pd.concat([red_m, red_u, red_2], axis=0)

timer.stop("9.2")

xs_red = normexp_get_xs(xf_red, controls=red_oob.values, offset=offset)

timer.stop("10")

cumsum = np.cumsum([0, len(red_m), len(red_u), len(red_2)])
range_red_m = range(cumsum[0], cumsum[1])
range_red_u = range(cumsum[1], cumsum[2])
range_red_2 = range(cumsum[2], cumsum[3])

timer.stop("11")

methyl = np.empty(self.methyl.shape)
unmethyl = np.empty(self.unmethyl.shape)

timer.stop("12")

methyl[:, i_grn_idx] = xs_grn["xs"][:, range_grn_m]
unmethyl[:, i_grn_idx] = xs_grn["xs"][:, range_grn_u]

timer.stop("13")

methyl[:, i_red_idx] = xs_red["xs"][:, range_red_m]
unmethyl[:, i_red_idx] = xs_red["xs"][:, range_red_u]

timer.stop("14")

methyl[:, ii_idx] = xs_grn["xs"][:, range_grn_2]
unmethyl[:, ii_idx] = xs_red["xs"][:, range_red_2]

timer.stop("15")

grn_control = grn.loc[control_probes.Address_ID]
red_control = red.loc[control_probes.Address_ID]

timer.stop("16")

xcs_grn = normexp_get_xs(grn_control, param=xs_grn["param"], offset=offset)
xcs_red = normexp_get_xs(red_control, param=xs_red["param"], offset=offset)

timer.stop("17.0")

cg_controls_idx = control_probes[
    control_probes.Control_Type.isin(["NORM_C", "NORM_G"])
].index
at_controls_idx = control_probes[
    control_probes.Control_Type.isin(["NORM_A", "NORM_T"])
].index

timer.stop("18")

grn_avg = np.mean(xcs_grn["xs"][:, cg_controls_idx], axis=1)
red_avg = np.mean(xcs_red["xs"][:, at_controls_idx], axis=1)

timer.stop("19")

red_grn_ratio = red_avg / grn_avg

timer.stop("20")

if dye_method == "single":
    red_factor = 1 / red_grn_ratio
    grn_factor = np.array([1, 1])
elif dye_method == "reference":
    ref_idx = np.argmin(np.abs(red_grn_ratio - 1))
    ref = (grn_avg + red_avg)[ref_idx] / 2
    if np.isnan(ref):
        raise ValueError("'ref_idx' refers to an array that is not present")
    grn_factor = ref / grn_avg
    red_factor = ref / red_avg

timer.stop("21")


methyl[:, i_red_idx] *= np.reshape(red_factor, (2, 1))
unmethyl[:, i_red_idx] *= np.reshape(red_factor, (2, 1))
unmethyl[:, ii_idx] *= np.reshape(red_factor, (2, 1))

timer.stop("22")

if dye_method == "reference":
    methyl[:, i_grn_idx] *= np.reshape(grn_factor, (2, 1))
    unmethyl[:, i_grn_idx] *= np.reshape(grn_factor, (2, 1))
    methyl[:, ii_idx] *= np.reshape(grn_factor, (2, 1))

timer.stop("23")

methyl_df = pd.DataFrame(
    methyl.T, index=self.methyl_ilmnid, columns=self.probes
)
unmethyl_df = pd.DataFrame(
    unmethyl.T, index=self.methyl_ilmnid, columns=self.probes
)

timer.stop("24")



# Noob
# >>> methyl_df
# 3999997083_R02C02  5775446049_R06C01
# cg13869341       18532.097447       31249.660200
# cg14008030        7381.869629       17687.621094
# cg12045430         123.438129         273.567078
# cg20826792         530.955529        1632.319576
# cg00381604         130.188501         247.995769
# ...                       ...                ...
# cg17939569          87.637659        5362.623047
# cg13365400          91.106116        5940.623047
# cg21106100          78.593774        7248.623047
# cg08265308         121.063723        5180.034367
# cg14273923          90.267410       10678.623047

# [485512 rows x 2 columns]
# >>> unmethyl_df
# 3999997083_R02C02  5775446049_R06C01
# cg13869341        1131.388267        4347.401849
# cg14008030        2242.245217       11188.378244
# cg12045430        7880.098972       15914.997346
# cg20826792        9914.292138       16576.082461
# cg00381604        8556.804514       13605.383527
# ...                       ...                ...
# cg17939569         110.009358        1331.076523
# cg13365400         114.410604        5418.527098
# cg21106100         117.836945         497.635410
# cg08265308         125.480597         263.723241
# cg14273923         149.288049        4527.317418


# Noob
# > head(M)
self.methylated.loc["cg00050873"] #          159.8342        17290.1654
self.methylated.loc["cg00212031"] #          137.8445          279.9552
self.methylated.loc["cg00213748"] #          137.6014         1669.8786
self.methylated.loc["cg00214611"] #          115.2945          257.6655
self.methylated.loc["cg00455876"] #          135.4455         4991.7508
self.methylated.loc["cg01707559"] #          136.6361          351.7144
self.methylated.loc["ch.22.44116734F"] #         213.94563          228.5676
self.methylated.loc["ch.22.909671F"] #           407.93938          286.5847
self.methylated.loc["ch.22.46830341F"] #         110.40962          361.7824
self.methylated.loc["ch.22.1008279F"] #           90.47575          208.9519
self.methylated.loc["ch.22.47579720R"] #         605.87659         1397.7026
self.methylated.loc["ch.22.48274842R"] #         249.49046         2775.6227

# > head(U)
self.unmethylated.loc["cg00050873"] #          131.0774          940.8142
self.unmethylated.loc["cg00212031"] #          130.8542         7701.6419
self.unmethylated.loc["cg00213748"] #          113.8862          291.8196
self.unmethylated.loc["cg00214611"] #          124.6565         7961.0551
self.unmethylated.loc["cg00455876"] #          183.1258          509.7173
self.unmethylated.loc["cg01707559"] #          128.8744         9980.5724
self.unmethylated.loc["ch.22.44116734F"] #          2724.296          5174.456
self.unmethylated.loc["ch.22.909671F"] #            1527.832          2677.953
self.unmethylated.loc["ch.22.46830341F"] #          6557.262         13366.890
self.unmethylated.loc["ch.22.1008279F"] #           2083.260          7672.353
self.unmethylated.loc["ch.22.47579720R"] #          7800.607         12882.931
self.unmethylated.loc["ch.22.48274842R"] #         13027.749         14065.632


