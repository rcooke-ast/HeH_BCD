
def GetAtomProp(line):
    prop = dict(line=line, wave=None, fval=None, lGamma=None, zabs=0.0)
    # H I lines
    if line == "HIb":
        prop['wave'], prop['fval'], prop['lGamma'] = 4862.691, 1.1938e-01, 13
    elif line == "HIg":
        prop['wave'], prop['fval'], prop['lGamma'] = 4341.691, 4.4694e-02, 13
    elif line == "HId":
        prop['wave'], prop['fval'], prop['lGamma'] = 4102.8991, 2.2105e-02, 13
    elif line == "HI7":
        prop['wave'], prop['fval'], prop['lGamma'] = 3971.198, 1.2711e-02, 13
    elif line == "HI8":
        prop['wave'], prop['fval'], prop['lGamma'] = 3890.166, 8.0397e-03, 13
    elif line == "HI9":
        prop['wave'], prop['fval'], prop['lGamma'] = 3836.485, 5.4317e-03, 13
    elif line == "HI10":
        prop['wave'], prop['fval'], prop['lGamma'] = 3798.987, 3.8526e-03, 13
    elif line == "HI11":
        prop['wave'], prop['fval'], prop['lGamma'] = 3771.704, 2.8368e-03, 13
    elif line == "HI12":
        prop['wave'], prop['fval'], prop['lGamma'] = 3751.217, 2.1521e-03, 13
    elif line == "HI13":
        prop['wave'], prop['fval'], prop['lGamma'] = 3735.431, 1.6728e-03, 13
    elif line == "HI14":
        prop['wave'], prop['fval'], prop['lGamma'] = 3723.005, 1.3269e-03, 13
    elif line == "HI15":
        prop['wave'], prop['fval'], prop['lGamma'] = 3713.034, 1.0708e-03, 13
    # He I lines
    elif line == "HeI3820":
        prop['wave'], prop['fval'], prop['lGamma'] = 3820.6914, 0.04694353, 1
    elif line == "HeI3889":
        prop['wave'], prop['fval'], prop['lGamma'] = 3889.750, 0.0644736, 1
    elif line == "HeI3965":
        prop['wave'], prop['fval'], prop['lGamma'] = 3965.8509, 4.9168e-02, 1
    elif line == "HeI4026":
        prop['wave'], prop['fval'], prop['lGamma'] = 4027.3292, 0.1, 1  # Actually, the f-value is 0.09402353
    elif line == "HeI4389":
        prop['wave'], prop['fval'], prop['lGamma'] = 4389.1624, 4.3269e-02, 1
    elif line == "HeI4472":
        prop['wave'], prop['fval'], prop['lGamma'] = 4472.7350, 0.2457056, 1
    else:
        print("No line found!!")
        assert(False)
    return prop
