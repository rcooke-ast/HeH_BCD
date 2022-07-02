
def GetAtomProp(line):
    prop = dict(line=line, wave=None, fval=None, lGamma=None, zabs=0.0)
    # H I lines
    if line == "HIb":
        prop['wave'], prop['fval'], prop['lGamma'] = 4862.691, 1.1938e-01, 13
    elif line == "HIg":
        prop['wave'], prop['fval'], prop['lGamma'] = 4341.691, 4.4694e-02, 13
    elif line == "HId":
        prop['wave'], prop['fval'], prop['lGamma'] = 4102.8991, 2.2105e-02, 13
    # He I lines
    elif line == "HeI4026":
        prop['wave'], prop['fval'], prop['lGamma'] = 4027.3292, 0.1, 1
    else:
        print("No line found!!")
        assert(False)
    return prop
