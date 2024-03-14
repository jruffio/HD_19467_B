import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.io.fits as fits
from glob import glob
import multiprocessing as mp

from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwstnirspec_cal import normalize_rows
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
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # External_dir should external files like the NIRCam filters
    external_dir = "/stow/jruffio/data/JWST/external/"
    # Science data: List of stage 2 cal.fits files
    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
    # utility folder where the intermediate and final data product will be saved
    utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils_fm/"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    # output dir for images
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
    ####################


    mypool = mp.Pool(processes=numthreads)

    filename =  filelist[1]

    if "nrs1" in filename:
        # continue
        detector = "nrs1"
        wv_sampling = wv_sampling_nrs1
        photfilter_name = photfilter_name_nrs1
        centroid = centroid_nrs1
        x_nodes = x_nodes_nrs1
    elif "nrs2" in filename:
        # continue
        detector = "nrs2"
        wv_sampling = wv_sampling_nrs2
        photfilter_name = photfilter_name_nrs2
        centroid = centroid_nrs2
        x_nodes = x_nodes_nrs2


    preproc_task_list = []
    preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
    preproc_task_list.append(["compute_coordinates_arrays"])
    preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
    preproc_task_list.append(["apply_coords_offset",{"coords_offset": centroid}])
    preproc_task_list.append(["compute_quick_webbpsf_model", {"image_mask": None, "pixelscale": 0.1, "oversample": 10}, True, True])
    # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
    preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}])
    preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": x_nodes, "mppool": mypool}, True, True])
    preproc_task_list.append(["compute_starsubtraction", {"mppool": mypool}, True, True])
    # preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, True])

    dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                              save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)

    plt.figure(1)
    plt.imshow(dataobj.data,interpolation="nearest",origin="lower")
    # plt.show()

    k=400
    threshold=10
    im_rows =dataobj.data
    im_wvs_rows = dataobj.wavelengths
    badpix_rows =dataobj.bad_pixels
    noise_rows = dataobj.noise
    star_model = np.ones(dataobj.data.shape)

    # plt.figure(2)
    # plt.plot(dataobj.data[400,:]*dataobj.bad_pixels[400,:],label="400")
    # plt.show()

    M_spline = get_spline_model(x_nodes, im_wvs_rows[k, :], spline_degree=3)


    where_data_finite = np.where(
        np.isfinite(badpix_rows[k, :]) * np.isfinite(im_rows[k, :]) * np.isfinite(noise_rows[k, :]) * (
                noise_rows[k, :] != 0) * np.isfinite(star_model[k, :]))

    d = im_rows[k, where_data_finite[0]]
    d_err = noise_rows[k, where_data_finite[0]]

    M = M_spline[where_data_finite[0], :] * star_model[k, where_data_finite[0], None]

    validpara = np.where(np.nansum(M > np.nanmax(M) * 0.01, axis=0) != 0)
    M = M[:, validpara[0]]

    # bounds_min = [0, ]* M.shape[1]
    bounds_min = [-np.inf, ] * M.shape[1]
    bounds_max = [np.inf, ] * M.shape[1]
    p = lsq_linear(M / d_err[:, None], d / d_err, bounds=(bounds_min, bounds_max)).x
    # p,chi2,rank,s = np.linalg.lstsq(M/d_err[:,None],d/d_err,rcond=None)
    m = np.dot(M, p)
    res = d - m
    new_im_rows = m
    new_noise_rows = d_err

    norm_res_row = np.zeros(im_rows.shape[1]) + np.nan
    norm_res_row[where_data_finite] = (d - m) / d_err
    meddev = median_abs_deviation(norm_res_row[where_data_finite])
    where_bad = np.where((np.abs(norm_res_row) / meddev > threshold) | np.isnan(norm_res_row))
    # new_badpix_rows[k, where_bad[0]] = np.nan


    fontsize = 12
    plt.figure(2,figsize=(12,12))
    plt.subplot(4, 1, 1)
    plt.title("The continuum of each pixel rows on the detector is fitted with a spline.",fontsize=fontsize)
    drow = np.zeros(im_wvs_rows[k,:].shape)+np.nan
    drow[where_data_finite] = d
    # plt.fill_between(im_wvs_rows[k,:],drow-noise_rows[k,:],drow+noise_rows[k,:],label="Error",alpha=0.5,color=color_list[1])
    plt.text(0.01, 0.99, "Detector NRS2 Row {0}".format(k), fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
    plt.plot(im_wvs_rows[k,:],drow*1e12,label="Flux",color=color_list[1])
    mrow = np.zeros(im_wvs_rows[k,:].shape)+np.nan
    mrow[where_data_finite] = m
    plt.plot(im_wvs_rows[k,:],mrow*1e12,label="Continuum (Spline)",color=color_list[0])
    for l in range(M.shape[1]):
        submrow = np.zeros(im_wvs_rows[k,:].shape)+np.nan
        submrow[where_data_finite] = M[:,l]
        if l==0:
            plt.plot(im_wvs_rows[k,:],submrow*p[l]*1e12,linestyle="--", color=color_list[2],alpha=0.5,label="Spline sub-components")
        else:
            plt.plot(im_wvs_rows[k,:],submrow*p[l]*1e12,linestyle="--", color=color_list[2],alpha=0.5)
    submrow = np.zeros(im_wvs_rows[k,:].shape)+np.nan
    submrow[where_data_finite] = M[:,10]
    plt.plot(im_wvs_rows[k,:],submrow*p[10]*1e12,color=color_list[2],label="Single sub-component")
    plt.fill_between([4.28,4.36],[-1e-9,-1e-9],[1e-9,1e-9],color="pink",alpha=0.4)
    plt.ylim([-1e2,8e2])
    plt.xlim([4.15,5.3])
    plt.plot(im_wvs_rows[k,:],(im_rows[k,:]-mrow)*1e12,color="grey",label="Residuals")
    # plt.ylim([-1e-10,1e-10])
    plt.legend(fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    # plt.xlabel("Wavelength ($\mu m$)",fontsize=fontsize)
    plt.ylabel("Flux ($\mu$Jy)",fontsize=fontsize)

    plt.subplot(4, 1, 2)
    plt.title("Each detector row is divided by its continuum.",fontsize=fontsize)
    # plt.plot(im_wvs_rows[k,:],drow/mrow,color="grey",linestyle="--",alpha=1,label="Continuum normalized row")
    plt.scatter(im_wvs_rows[k,:],drow/mrow,color="black",alpha=1,s=20,label="Continuum normalized row", linewidth=0)
    plt.fill_between(im_wvs_rows[k,:],1-noise_rows[k,:]/mrow,1+noise_rows[k,:]/mrow,label="Scaled error amplitude",alpha=0.5,color=color_list[1], lw=0)
    plt.fill_between([4.28,4.36],[0,0],[2,2],color="pink",alpha=0.4,label="Region highlighed in panel 3")
    plt.ylim([0.8,1.2])
    plt.xlim([4.15,5.3])
    plt.legend(fontsize=fontsize,loc="upper left")
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    # plt.xlabel("Wavelength ($\mu m$)",fontsize=fontsize)
    plt.ylabel("Normalized Flux",fontsize=fontsize)
    # plt.ylim([-1e-10,1e-10])

    new_wavelengths, combined_fluxes, combined_errors, spline_cont0,spline_paras0,_ = dataobj.reload_starspectrum_contnorm()
    err = dataobj.noise
    spline_cont0[np.where(spline_cont0 / err < 5)] = np.nan
    spline_cont0 = copy(spline_cont0)
    spline_cont0[np.where(spline_cont0 < np.median(spline_cont0))] = np.nan
    spline_cont0[np.where(np.isnan(dataobj.bad_pixels))] = np.nan
    normalized_im = dataobj.data / spline_cont0
    normalized_err = err / spline_cont0



    plt.subplot(4, 1, 3)
    plt.title("The continuum normalized fluxes from the entire image are combined in each wavelength bin.",fontsize=fontsize)
    withinbounds = np.where((dataobj.wavelengths>4.28)*(dataobj.wavelengths<4.36))
    # import random
    # samples = random.sample(range(0, np.size(withinbounds[0]) + 1), 10000)
    # plt.scatter(dataobj.wavelengths[withinbounds][samples],normalized_im[withinbounds][samples],color="black",alpha=0.5,s=1,label="Continuum normalized image")
    plt.scatter(dataobj.wavelengths[withinbounds],normalized_im[withinbounds],color="black",alpha=0.5,s=1,label="Continuum normalized image", linewidth=0)
    plt.plot(new_wavelengths,combined_fluxes,label="Combined spectrum",color=color_list[0])
    plt.fill_between([4.28,4.36],[0,0],[2,2],color="pink",alpha=0.4)
    plt.xlim([4.28,4.36])
    plt.ylim([0.8,1.2])
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    # plt.xlabel("Wavelength ($\mu m$)",fontsize=fontsize)
    plt.ylabel("Normalized Flux",fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.subplot(4, 1, 4)
    plt.title("A final high S/N continuum normalized spectrum is obtained.",fontsize=fontsize)
    plt.plot(new_wavelengths,combined_fluxes,label="Combined spectrum",color=color_list[0])
    plt.fill_between([4.28,4.36],[0,0],[2,2],color="pink",alpha=0.4)
    plt.xlim([4.15,5.3])
    plt.ylim([0.93,1.05])
    plt.legend(fontsize=fontsize, loc="upper left")
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)

    plt.xlabel("Wavelength ($\mu m$)",fontsize=fontsize)
    plt.ylabel("Normalized Flux",fontsize=fontsize)
    plt.tight_layout()
    out_filename = os.path.join(out_png,"contnormspec.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename,dpi=300)
    plt.savefig(out_filename.replace(".png",".pdf"))

    plt.show()
