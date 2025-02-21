from PseudoAAC import GetAAComposition

def AAC(seq):
    code = GetAAComposition(seq)
    res = []
    for v in code.values():
        res.append(v)
    return res