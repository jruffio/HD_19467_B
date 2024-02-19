import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import scipy.io as scio
import astropy.io.fits as fits
import pandas as pd
from glob import glob
import multiprocessing as mp
import h5py
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const
from scipy.signal import correlate2d

from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.grid_search import grid_search
from breads.fm.hc_splinefm_jwst_nirspec import hc_splinefm_jwst_nirspec
from scipy.stats import median_abs_deviation

from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')
from breads.utils import get_spline_model
from scipy.optimize import lsq_linear

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass
    ####################
    ## To be modified
    ####################
    # Number of threads to be used for multithreading
    numthreads = 20
    # Number of nodes
    nodes = 40
    # Number of principal components (Karhunen-Loeve) modes to be added to the forward model
    N_KL = 3#0#3
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # External_dir should external files like the NIRCam filters
    external_dir = "/stow/jruffio/data/JWST/external/"
    # Science data: List of stage 2 cal.fits files
    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs2_cal.fits")
    filelist.sort()
    filename = filelist[0]
    print(filename)
    # utility folder where the intermediate and final data product will be saved
    utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils/"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"

    color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
    ####################
    ## No need to change for HD19467B prog. id 1414
    # NIrcam filter used for flux normalization
    photfilter_name_nrs1 =  "F360M"
    photfilter_name_nrs2 =  "F460M"
    # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
    wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
    wv_sampling_nrs2 = np.arange(4.081285, 5.278689, 0.0006656647)
    # definiton of the nodes for the spline
    x_nodes_nrs1 = np.linspace(2.859509, 4.1012874, nodes, endpoint=True)
    x_nodes_nrs2 = np.linspace(4.081285, 5.278689, nodes, endpoint=True)
    # List of empirical offsets to correct the wcs coordinate system
    centroid_nrs1 = [-0.13499443, -0.07202978]
    centroid_nrs2 = [-0.12986898, -0.08441719]


    if "nrs1" in filename:
        detector = "nrs1"
        wv_sampling = wv_sampling_nrs1
        photfilter_name = photfilter_name_nrs1
        centroid = centroid_nrs1
        x_nodes = x_nodes_nrs1
    elif "nrs2" in filename:
        detector = "nrs2"
        wv_sampling = wv_sampling_nrs2
        photfilter_name = photfilter_name_nrs2
        centroid = centroid_nrs2
        x_nodes = x_nodes_nrs2


    mypool = mp.Pool(processes=numthreads)

    preproc_task_list = []
    preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
    preproc_task_list.append(["compute_coordinates_arrays"])
    preproc_task_list.append(["convert_MJy_per_sr_to_MJy"]) # old reduction, already in MJy
    preproc_task_list.append(["apply_coords_offset",{"coords_offset":(centroid[0],centroid[1])}]) #-0.11584366936455087, 0.07189009712128012
    # preproc_task_list.append(["compute_webbpsf_model",
    #                           {"wv_sampling": wv_sampling, "image_mask": None, "pixelscale": 0.1, "oversample": 10,
    #                            "parallelize": False, "mppool": mypool}, True, True])
    # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
    preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}, True, True])
    preproc_task_list.append(["compute_starspectrum_contnorm", {"mppool": mypool}, True, True])
    preproc_task_list.append(["compute_starsubtraction",{"mppool":mypool},True,True])
    preproc_task_list.append(["compute_interpdata_regwvs",{"wv_sampling":wv_sampling},True,True])

    dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                              save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)

    interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = dataobj.reload_interpdata_regwvs()

    fontsize = 12
    plt.figure(1,figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(dataobj.star_func(dataobj.wavelengths)*dataobj.bad_pixels,interpolation="nearest",origin="lower")
    # plt.gca().set_aspect('equal')
    plt.text(0.03, 0.99, "Original", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes)
    plt.clim([0.96,1.04])
    plt.xlabel("x-axis (pix)",fontsize=fontsize)
    plt.ylabel("y-axis (pix)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    # plt.show()

    plt.subplot(1,3,2)
    dwv = wv_sampling[1]-wv_sampling[0]
    wv0 = 4.7
    plt.imshow(dataobj.star_func(interp_wvs)*interp_badpix,interpolation="nearest",origin="lower",extent=[wv_sampling[0]-dwv/2.,wv_sampling[-1]+dwv/2,0-0.5,interp_wvs.shape[0]+0.5])
    plt.plot([wv0,wv0],[-0.5,interp_wvs.shape[0]+0.5],color=color_list[0],linestyle="--",linewidth=2)
    plt.gca().set_aspect((wv_sampling[-1]-wv_sampling[0])/interp_wvs.shape[0])
    plt.text(0.03, 0.99, "Interpolated", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
    plt.text(wv0+0.01,0.0, "Cut", fontsize=fontsize, ha='left', va='bottom', color=color_list[0])
    plt.clim([0.96,1.04])
    plt.xlabel("Wavelength ($\mu m$)",fontsize=fontsize)
    plt.ylabel("y-axis (pix)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)


    l = np.argmin(np.abs(wv_sampling-wv0))

    plt.subplot(1,3,3)
    plt.scatter(interp_ra[:,l]*interp_badpix[:,l], interp_dec[:,l]*interp_badpix[:,l], s=5, c=color_list[0])
    plt.xlim([-2.5,2])
    plt.ylim([-3+0.3,1.5+0.3])
    plt.gca().set_aspect('equal')
    plt.text(0.03, 0.99, "Spatial sampling of cut", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
    plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
    plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.gca().invert_xaxis()


    plt.tight_layout()
    out_filename = os.path.join(out_png,"regwvsinterp.png")
    print("Saving " + out_filename)
    # plt.savefig(out_filename,dpi=300)
    # plt.savefig(out_filename.replace(".png",".pdf"))

    plt.show()
