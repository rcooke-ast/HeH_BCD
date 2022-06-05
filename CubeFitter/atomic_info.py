
def GetAtomProp(line):
    prop = dict(wave=None, fval=None, lGamma=None)
    if line == "HIb":
        prop['wave'], prop['fval'], prop['lGamma'] = 4862.691, 1.1938e-01, 9
    elif line == "HIg":
        prop['wave'], prop['fval'], prop['lGamma'] = 4341.691, 4.4694e-02, 9
    elif line == "HId":
        prop['wave'], prop['fval'], prop['lGamma'] = 4102.8991, 2.2105e-02, 9
    else:
        print("No line found!!")
        assert(False)
    return prop
