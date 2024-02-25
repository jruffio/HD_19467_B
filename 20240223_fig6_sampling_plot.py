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

from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')
from breads.instruments.jwstnirspec_cal import rotate_coordinates,mycostfunc
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize
import matplotlib.patheffects as PathEffects

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    # If True, make the size of the points proportional to the flux
    scale_scatter_vs_flux = False
    detector = "nrs2"
    # Wavelengths to plots
    wv0 = 4.25
    wv1 = 5.0

    color_list = ["#006699","#ff9900", "#6600ff", "#006699", "#ff9900", "#6600ff"]
    # output dir for images
    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"

    if 1:
        ####################
        ## To be modified
        ####################
        # Number of threads to be used for multithreading
        numthreads = 20
        # Number of nodes
        nodes = 40
        # Directories to update
        os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
        crds_dir="/stow/jruffio/data/JWST/crds_cache/"
        # Suffix to be added to the output filename that contained the extracted spectrum and centroid
        filename_suffix = "_webbpsf"
        # List of stage 2 cal.fits files to be extracted
        filelist = glob("/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/20240124_stage2_clean/jw01128009001_03108_*_nrs*_cal.fits")
        filelist.sort()
        for filename in filelist:
            print(filename)
        print("N files: {0}".format(len(filelist)))
        # utility folder where the intermediate and final data product will be saved
        utils_dir = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/breads/20240127_utils_clean/"
        if not os.path.exists(utils_dir):
            os.makedirs(utils_dir)
        ####################
        ## No need to change for TYC_4433-1800-1 prog. id 1128
        # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
        wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
        wv_sampling_nrs2 = np.arange(4.081285, 5.278689, 0.0006656647)
        # offset of the coordinates because point cloud is not centered
        init_centroid_nrs1 = [-0.25973700664819993, 0.7535417070247359]
        init_centroid_nrs2 = [-0.2679950725373308, 0.7554649479920329]
        # Flux Calibration parameters
        flux_calib_paras = -0.03864459,  1.09360589
        ####################

        # Define a multiprocessing pool for multi-threading when needed below
        mypool = mp.Pool(processes=numthreads)

        #################################
        # Looping over both NIRSpec detectors
        if detector == "nrs1":
            init_centroid = init_centroid_nrs1
            wv_sampling = wv_sampling_nrs1
        elif detector == "nrs2":
            init_centroid = init_centroid_nrs2
            wv_sampling = wv_sampling_nrs2

        # Select only the files corresponding to the correct detector
        nrs_filelist = []
        for filename in filelist:
            if detector in filename:
                nrs_filelist.append(filename)
        nrs_filelist.sort()

        # Definie the filename of the output file saved by fitpsf
        splitbasename = os.path.basename(nrs_filelist[0]).split("_")
        fitpsf_filename = os.path.join(utils_dir, splitbasename[0] + "_" + splitbasename[1] + "_" + splitbasename[3] + "_fitpsf" + filename_suffix + ".fits")

        # List of preprocessing steps to be applied to each cal.fits file
        # It's possible to call each step directly as dataobj.compute_xyz(parameters,..) if you prefer.
        # This is only for convenience.
        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays", {}, True, True])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])  # old reduction, already in MJy
        preproc_task_list.append(["apply_coords_offset",{"coords_offset":init_centroid}])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"mppool": mypool}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"mppool": mypool,"threshold_badpix": 50}, True, False])
        preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, False])

        # Make a list of all the data objects to be given to fit psf
        dataobj_list = []
        for filename in nrs_filelist:
            print(filename)
            dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                      save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)

            dataobj_list.append(dataobj)

        dataobj0 = dataobj_list[0]

        reload_interpdata_outputs = dataobj0.reload_interpdata_regwvs(load_filename=dataobj0.default_filenames["compute_interpdata_regwvs"])
        if reload_interpdata_outputs is None:
            reload_interpdata_outputs = dataobj0.compute_interpdata_regwvs(save_utils=True)
        all_interp_ra, all_interp_dec, all_interp_wvs, all_interp_flux, all_interp_err, all_interp_badpix, all_interp_area2d = reload_interpdata_outputs

        wv_sampling = dataobj0.wv_sampling
        if len(dataobj_list) > 1:
            for dataobj in dataobj_list[1::]:
                reload_interpdata_outputs = dataobj.reload_interpdata_regwvs(
                    load_filename=dataobj.default_filenames["compute_interpdata_regwvs"])
                if reload_interpdata_outputs is None:
                    reload_interpdata_outputs = dataobj.compute_interpdata_regwvs(save_utils=True)
                interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = reload_interpdata_outputs

                all_interp_ra = np.concatenate((all_interp_ra, interp_ra), axis=0)
                all_interp_dec = np.concatenate((all_interp_dec, interp_dec), axis=0)
                all_interp_wvs = np.concatenate((all_interp_wvs, interp_wvs), axis=0)
                all_interp_flux = np.concatenate((all_interp_flux, interp_flux), axis=0)
                all_interp_err = np.concatenate((all_interp_err, interp_err), axis=0)
                all_interp_badpix = np.concatenate((all_interp_badpix, interp_badpix), axis=0)
                all_interp_area2d = np.concatenate((all_interp_area2d, interp_area2d), axis=0)

        l0 = np.argmin(np.abs(wv_sampling-wv0))
        l1 = np.argmin(np.abs(wv_sampling-wv1))

        fontsize = 12
        plt.figure(1,figsize=(12,12))
        for lid,(wv,l) in enumerate(zip([wv0,wv1],[l0,l1])):
            plt.subplot(2,2,1+lid)
            if scale_scatter_vs_flux:
                plt.scatter(all_interp_ra[:,l],all_interp_dec[:,l],s=10*all_interp_flux[:,l]/np.nanmax(all_interp_flux[:,l]),c=color_list[lid])
            else:
                plt.scatter(all_interp_ra[:,l],all_interp_dec[:,l],s=1,c=color_list[lid])
            txt = plt.text(0.03, 0.99, "4 dithers - TYC 4433-1800-1 - $\lambda$={0} $\mu$m".format(wv), fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
            # cbar = plt.colorbar(pad=0)#mappable=CS,
            # cbar.ax.tick_params(labelsize=fontsize)
            # cbar.set_label(label='Flux (MJy)', fontsize=fontsize)
            # plt.clim([0, 0.5e-9])
            # plt.legend(loc="lower left",fontsize=fontsize)
            rad = 1.5
            for x in np.arange(-rad,rad,0.1):
                plt.plot([x,x],[-rad,rad],color="grey",linewidth=1,alpha=0.5)
                plt.plot([-rad,rad],[x,x],color="grey",linewidth=1,alpha=0.5)
            plt.xlim([-rad,rad])
            plt.ylim([-rad,rad])
            plt.gca().set_aspect('equal')
            plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
            plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.gca().invert_xaxis()

        # plt.show()

    if 1:  # science
        ####################
        ## To be modified
        ####################
        # Number of threads to be used for multithreading
        numthreads = 20
        # Number of nodes
        nodes = 40
        # Directories to update
        os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
        crds_dir="/stow/jruffio/data/JWST/crds_cache/"
        if 1: # Science data: List of stage 2 cal.fits files
            filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
            filelist.sort()
            for filename in filelist:
                print(filename)
            print("N files: {0}".format(len(filelist)))
            # utility folder where the intermediate and final data product will be saved
            utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils/"
            if not os.path.exists(utils_dir):
                os.makedirs(utils_dir)
        ####################
        ## No need to change for HD19467B prog. id 1414
        # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
        wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
        wv_sampling_nrs2 = np.arange(4.081285, 5.278689, 0.0006656647)
        # List of empirical offsets to correct the wcs coordinate system
        sc_centroid_nrs1 = [-0.13499443, -0.07202978]
        sc_centroid_nrs2 = [-0.12986898, -0.08441719]
        ####################
        if detector == "nrs1":
            wv_sampling = wv_sampling_nrs1
            sc_centroid = sc_centroid_nrs1
        elif detector == "nrs2":
            wv_sampling = wv_sampling_nrs2
            sc_centroid = sc_centroid_nrs2

        # Select only the files corresponding to the correct detector
        nrs_filelist = []
        for filename in filelist:
            if detector in filename:
                nrs_filelist.append(filename)
        nrs_filelist.sort()

        # List of preprocessing steps to be applied to each cal.fits file
        # It's possible to call each step directly as dataobj.compute_xyz(parameters,..) if you prefer.
        # This is only for convenience.
        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays"])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"]) # old reduction, already in MJy
        preproc_task_list.append(["apply_coords_offset",{"coords_offset":(sc_centroid[0],sc_centroid[1])}])
        # preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}, True, True])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"mppool": mypool}, True, True])
        preproc_task_list.append(["compute_starsubtraction",{"mppool":mypool},True,True])
        preproc_task_list.append(["compute_interpdata_regwvs",{"wv_sampling":wv_sampling},True,True])

        dataobj_list = []
        # cen_list = []
        for filename in nrs_filelist:
            print(filename)
            dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                      save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
            dataobj_list.append(dataobj)

        dataobj0 = dataobj_list[0]

        reload_interpdata_outputs = dataobj0.reload_interpdata_regwvs(load_filename=dataobj0.default_filenames["compute_interpdata_regwvs"])
        if reload_interpdata_outputs is None:
            reload_interpdata_outputs = dataobj0.compute_interpdata_regwvs(save_utils=True)
        all_interp_ra, all_interp_dec, all_interp_wvs, all_interp_flux, all_interp_err, all_interp_badpix, all_interp_area2d = reload_interpdata_outputs

        wv_sampling = dataobj0.wv_sampling
        if len(dataobj_list) > 1:
            for dataobj in dataobj_list[1::]:
                reload_interpdata_outputs = dataobj.reload_interpdata_regwvs(
                    load_filename=dataobj.default_filenames["compute_interpdata_regwvs"])
                if reload_interpdata_outputs is None:
                    reload_interpdata_outputs = dataobj.compute_interpdata_regwvs(save_utils=True)
                interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = reload_interpdata_outputs

                all_interp_ra = np.concatenate((all_interp_ra, interp_ra), axis=0)
                all_interp_dec = np.concatenate((all_interp_dec, interp_dec), axis=0)
                all_interp_wvs = np.concatenate((all_interp_wvs, interp_wvs), axis=0)
                all_interp_flux = np.concatenate((all_interp_flux, interp_flux), axis=0)
                all_interp_err = np.concatenate((all_interp_err, interp_err), axis=0)
                all_interp_badpix = np.concatenate((all_interp_badpix, interp_badpix), axis=0)
                all_interp_area2d = np.concatenate((all_interp_area2d, interp_area2d), axis=0)

        l0 = np.argmin(np.abs(wv_sampling-wv0))
        l1 = np.argmin(np.abs(wv_sampling-wv1))

        for lid,(wv,l) in enumerate(zip([wv0,wv1],[l0,l1])):
            plt.subplot(2,2,3+lid)
            if scale_scatter_vs_flux:
                plt.scatter(all_interp_ra[:,l],all_interp_dec[:,l],s=10*all_interp_flux[:,l]/np.nanmax(all_interp_flux[:,l]),c=color_list[lid])
            else:
                plt.scatter(all_interp_ra[:,l],all_interp_dec[:,l],s=1,c=color_list[lid])
            txt = plt.text(0.03, 0.99, "9 dithers - HD 19467 - $\lambda$={0} $\mu$m".format(wv), fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
            # cbar = plt.colorbar(pad=0)#mappable=CS,
            # cbar.ax.tick_params(labelsize=fontsize)
            # cbar.set_label(label='Flux (MJy)', fontsize=fontsize)
            # plt.clim([0, 0.5e-9])
            # plt.legend(loc="lower left",fontsize=fontsize)
            for x in np.arange(-rad,rad,0.1):
                plt.plot([x,x],[-rad,rad],color="grey",linewidth=1,alpha=0.5)
                plt.plot([-rad,rad],[x,x],color="grey",linewidth=1,alpha=0.5)
            plt.xlim([-rad,rad])
            plt.ylim([-rad,rad])
            plt.gca().set_aspect('equal')
            plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
            plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.gca().invert_xaxis()





    plt.tight_layout()
    out_filename = os.path.join(out_png, "sampling.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))


    plt.show()
