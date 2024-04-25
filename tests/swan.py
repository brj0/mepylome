
raw = RawData([ref0, ref1])

timer.start()
self = MethylData(raw, prep="swan")
timer.stop("1")


raw = RawData([ref0, ref1])
meth = MethylData(raw, prep="swan")
M_s = meth.methylated
U_s = meth.unmethylated
for _ in range(999):
    print(_)
    # meth = MethylData(raw)
    meth = MethylData(raw, prep="swan")
    M_s += meth.methylated
    U_s += meth.unmethylated

M = M_s / 1000
U = U_s / 1000



