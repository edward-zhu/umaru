import random
N = 1000000
wt = [10, 20, 40]
wtp = [1.*x/sum(wt) for x in wt]
result = []
p = [random.normalvariate(1./x, 1./x/3.) for x in wtp]
for i in xrange(N):
    minp = 1.e9
    minj = -1
    for j, pp in enumerate(p):
        if pp < minp:
            minp = pp
            minj = j
    result.append(minj)
    for j, pp in enumerate(p):
        p[j] -= minp
    p[minj] = random.normalvariate(1./wtp[minj], 1./wtp[minj]/3.)
    
