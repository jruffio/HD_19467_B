
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # WARNING: This script assumes purely photon noise limited observations.
    # We do account for any systematic floor, saturation, thermal background etc.

    myfilter= "F444W" #Choices of: F356W, F444W, Lp, Mp
    myTeff = 500 #Choices of: 500, 1000, 1500, 2000, 2500, 3000 (K)
    Tint = 35 # in minutes
    star_Kmag = 5.4 # apparent magnitude of the considered star in K band(Vega mag)
    fontsize = 12

    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
    contrasts_filename = os.path.join(out_png, "HD19467B_5sig_contrast_vs_Teff_allfilters.csv")

    # Load the CSV file into a NumPy array, skipping comment lines
    np_array = np.genfromtxt(contrasts_filename, delimiter='\t', comments='#',dtype=str)
    # File comments:
    # Ruffio et al. 2023 (in prep.)
    # Detection sensitivity of NIRSpec as a function of effective temperature of the companion model (BTSettl - Allard+2003)
    # GTO 1414 - HD 19467 - Kmag=5.4 - Tint=35 min (no overheads included)
    # (1st column) Separation to the star in milli-arcsecond
    # (other columns) 5sigma detection limits of NIRSpec IFU expressed as a companion-to-star flux ratio in the indicated filter
    # Filter reference: http://svo2.cab.inta-csic.es/theory/fps/
    # F356W = JWST_NIRCam.F356W.dat
    # F444W = JWST_NIRCam.F444W.dat
    # Lp = Keck_NIRC2.Lp.dat
    # Mp = Paranal_NACO.Mp.dat
    HD19467B_Tint = 35 # min
    HD19467B_Kmag = 5.4 # Vega mag


    # Read the column labels
    labels = np_array[0,:].tolist()
    # Read the vector of separations in mas
    contrast_5sig_1D_seps = np_array[1::,0].astype(float)/1000 # convert from mas to as
    # Read the table of 5 sigma contrast curves for all combinations of Teff and filter
    contrast_5sig_1D_med_array_filters = np_array[1::,1::].astype(float)
    # select the right column as a function of Teff and the filter
    column_id = labels.index('{0} - Teff = {1}K'.format(myfilter,myTeff))

    # scale the contrast with respect to exposure time and stellar apparent magnitude
    scaled_5sig_contrast = contrast_5sig_1D_med_array_filters[:, column_id]*np.sqrt(HD19467B_Tint/Tint*10**((star_Kmag-HD19467B_Kmag)/2.5))

    plt.plot(contrast_5sig_1D_seps,scaled_5sig_contrast,alpha=1,c="orange",linestyle="-")
    plt.text(0.02, 0.02, r"Kmag={0}".format(star_Kmag)+" - $T_{\mathrm{int}}=$"+"{0} min".format(Tint), fontsize=fontsize, ha='left', va='bottom', transform=plt.gca().transAxes,color="Black")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([0.1,3])
    plt.ylim([1*10**(-7),10**(-3)])
    plt.fill_between([0,0.3],[10**(-10),10**(-10)],[1,1],color="grey",alpha=0.2)
    plt.xlabel("Separation (as)",fontsize=fontsize)
    plt.ylabel("5$\sigma$ Flux ratio ({0})".format(myfilter),fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.show()

