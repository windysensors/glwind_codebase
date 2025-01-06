### stat_calc.py ###
# author: Elliott Walker
# last update: 9 July 2024
# description: Functions for statistical calculations

# least squares linear fit - fits data in (x,y) pairs to a relationship y = A + B*x
# xvals and yvals should each be iterables of the same length
# outputs pair A,B
def ls_linear_fit(xvals, yvals):
    if len(yvals) == 0 or len(xvals) == 0:
        return 0,0
    xvals = list(xvals)
    yvals = list(yvals)
    if len(yvals) != len(xvals):
        throw("Lists must be of equal size")
    for x, y in zip(xvals, yvals):
        if math.isnan(y):
            xvals.remove(x)
            yvals.remove(y)
    n = len(xvals)
    sum_x = sum(xvals)
    sum_x2 = sum(x*x for x in xvals)
    sum_xy = sum(xvals[i]*yvals[i] for i in range(n))
    sum_y = sum(yvals)
    det = n * sum_x2 - sum_x * sum_x
    A = (sum_y * sum_x2 - sum_x * sum_xy)/det
    B = (n * sum_xy - sum_x * sum_y)/det
    return A, B

# power law fit - fits data in (x,y) pairs to a relationship y = A*x**B
# xvals and yvals should each be iterables of the same length
# pass argument both=True to obtain both A and B
#   otherwise only the exponent B will be returned (wind shear exponent)
# outputs either just B or pair A,B based on "both"
def power_fit(xvals, yvals, both=False):
    xconsider = []
    yconsider = []
    for x,y in zip(xvals, yvals):
        if not (math.isnan(x) or math.isnan(y)):
            xconsider.append(x)
            yconsider.append(y)
    lnA, B = ls_linear_fit(np.log(xconsider),np.log(yconsider))
    if both:
        return np.exp(lnA), B
    return B

# angle average - finds the angle `between` two other angles, expressed in degrees
# if the angles are exactly opposite (180 degrees apart) the result will be an angular increase of alpha
def angle_average(alpha, beta):
    minimal_difference = ((beta - alpha + 180) % 360) - 180
    avg = (alpha + minimal_difference/2) % 360
    return avg
