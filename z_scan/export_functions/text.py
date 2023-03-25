def best(fit_type, chi2_best, p_best):
    # 1PA
    if fit_type == 0:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'Is1 = %.2e W/mm2' % (p_best[1],)))

    # 2PA no Is2
    elif fit_type == 1:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'Is1 = %.2e W/mm2' % (p_best[1],),
        r'beta = %.2e mm/W' % (p_best[2],)))

    # 2PA
    elif fit_type == 2:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'Is1 = %.2e W/mm2' % (p_best[1],),
        r'Is2 = %.2e W/mm2' % (p_best[2],),
        r'beta = %.2e mm/W' % (p_best[3],)))
            
    # 2PA no Is1
    elif fit_type == 3:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'Is2 = %.2e W/mm2' % (p_best[1],),
        r'beta = %.2e mm/W' % (p_best[2],)))

    # 2PA no sat
    else:
        textstr = '\n'.join((
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = %.2f mm' % (p_best[0], ),
        r'beta = %.2e mm/W' % (p_best[1],)))
    return textstr

##############################################################

def individual(fit_type, runs, i):
    # 1PA
    if fit_type == 0:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,2], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'Is1 = %.2e W/mm2' % (runs[i,1],)))

    # 2PA no Is2
    elif fit_type == 1:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,3], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'Is1 = %.2e W/mm2' % (runs[i,1],),
        r'beta = %.2e mm/W' % (runs[i,2],)))

    # 2PA
    elif fit_type == 2:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,4], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'Is1 = %.2e W/mm2' % (runs[i,1],),
        r'Is2 = %.2e W/mm2' % (runs[i,2],),
        r'beta = %.2e mm/W' % (runs[i,3],)))
            
    # 2PA no Is1
    elif fit_type == 3:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,3], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'Is2 = %.2e W/mm2' % (runs[i,1],),
        r'beta = %.2e mm/W' % (runs[i,2],)))

    # 2PA no sat
    else:
        textstr = '\n'.join((
        r'X2 = %.2e' % (runs[i,2], ),
        r'z0 = %.2f mm' % (runs[i,0], ),
        r'beta = %.2e mm/W' % (runs[1],)))
    return textstr

###########################################################################

def output(fit_type, chi2_best, p_best, sigma, span):
    # 1PA
    if fit_type == 0:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'Is1 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[1], sigma[1], span[1],)))

    # 2PA no Is2
    elif fit_type == 1:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'Is1 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[1], sigma[1], span[1], ), 
        r'beta = (%.2e +- %.1e)  mm/W; X2 span: %.2f' % (p_best[2], sigma[2], span[2],)))

    # 2PA
    elif fit_type == 2:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'Is1 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[1], sigma[1], span[1], ),
        r'Is2 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[2], sigma[2], span[2], ),
        r'beta = (%.2e +- %.1e)  mm/W; X2 span: %.2f' % (p_best[3], sigma[3], span[3],)))
            
    # 2PA no Is1
    elif fit_type == 3:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'Is2 = (%.2e +- %.1e)  W/mm2; X2 span: %.2f' % (p_best[1], sigma[1], span[1], ),
        r'beta = (%.2e +- %.1e)  mm/W; X2 span: %.2f' % (p_best[2], sigma[2], span[2],)))

    # 2PA no sat
    else:
        textstr = '\n'.join((
        r'Fitting Parameters',
        r'X2 = %.2e' % (chi2_best, ),
        r'z0 = (%.2f +- %.2f) mm; X2 span: %.2f' % (p_best[0], sigma[0], span[0], ),
        r'beta = (%.2e +- %.1e)  mm/W; X2 span: %.2f' % (p_best[1], sigma[1], span[1],)))
    return textstr