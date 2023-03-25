import matplotlib.pyplot as plt
from z_scan.fitting_model.absorption import transmittance, intensity

def Tz(ax: plt.Axes, z_data, I_data, z_plot, sigma, fit_type, p_best, experiment_param):
    """Plots data and T(z) based on fit results"""

    # 1PA
    if fit_type == 0:
        ax.errorbar(z_data, I_data, sigma, fmt='o')
        ax.plot(z_plot, transmittance.OPA(z_plot, *p_best, *experiment_param))

    # 2PA no Is2
    elif fit_type == 1:
        ax.errorbar(z_data, I_data, sigma, fmt='o')
        ax.plot(z_plot, transmittance.TPA_no_Is2(z_plot, *p_best, *experiment_param))

    # 2PA
    elif fit_type == 2:
        ax.errorbar(z_data, I_data, sigma, fmt='o')
        ax.plot(z_plot, transmittance.TPA(z_plot, *p_best, *experiment_param))

    # 2PA no Is1
    elif fit_type == 3:
        ax.errorbar(z_data, I_data, sigma, fmt='o')
        ax.plot(z_plot, transmittance.TPA_no_Is1(z_plot, *p_best, *experiment_param))

    # 2PA no sat
    else:
        ax.errorbar(z_data, I_data, sigma, fmt='o')
        ax.plot(z_plot, transmittance.TPA_no_sat(z_plot, *p_best, *experiment_param))
    return

##############################################################################################

def TI(ax: plt.Axes, z_data, I_data, z_plot, sigma, fit_type, p_best, experiment_param):
    # 1PA
    if fit_type == 0:
        ax.errorbar(intensity(z_data, p_best[0], experiment_param[2], experiment_param[3]), I_data, sigma, fmt='o')
        ax.plot(intensity(z_plot, p_best[0], experiment_param[2], experiment_param[3]), transmittance.OPA(z_plot, *p_best, *experiment_param))
        ax.axvline(p_best[1], ls=':', color='green', label=r'$I_{s1}$')

    # 2PA no Is2
    elif fit_type == 1:
        ax.errorbar(intensity(z_data, p_best[0], experiment_param[2], experiment_param[3]), I_data, sigma, fmt='o')
        ax.plot(intensity(z_plot, p_best[0], experiment_param[2], experiment_param[3]), transmittance.TPA_no_Is2(z_plot, *p_best, *experiment_param))
        ax.axvline(p_best[1], ls=':', color='green', label=r'$I_{s1}$')

    # 2PA
    elif fit_type == 2:
        ax.errorbar(intensity(z_data, p_best[0], experiment_param[2], experiment_param[3]), I_data, sigma, fmt='o')
        ax.plot(intensity(z_plot, p_best[0], experiment_param[2], experiment_param[3]), transmittance.TPA(z_plot, *p_best, *experiment_param))
        ax.axvline(p_best[1], ls=':', color='green', label=r'$I_{s1}$')
        ax.axvline(experiment_param[1]/p_best[-1], ls=':', color='red', label=r'$\alpha / \beta$')

    # 2PA no Is1
    elif fit_type == 3:
        ax.errorbar(intensity(z_data, p_best[0], experiment_param[2], experiment_param[3]), I_data, sigma, fmt='o')
        ax.plot(intensity(z_plot, p_best[0], experiment_param[2], experiment_param[3]), transmittance.TPA_no_Is1(z_plot, *p_best, *experiment_param))
        ax.axvline(experiment_param[1]/p_best[-1], ls=':', color='red', label=r'$\alpha / \beta$')
    
    # 2PA no sat
    else:
        ax.errorbar(intensity(z_data, p_best[0], experiment_param[2], experiment_param[3]), I_data, sigma, fmt='o')
        ax.plot(intensity(z_plot, p_best[0], experiment_param[2], experiment_param[3]), transmittance.TPA_no_sat(z_plot, *p_best, *experiment_param))
        ax.axvline(experiment_param[1]/p_best[-1], ls=':', color='red', label=r'$\alpha / \beta$')
    return
