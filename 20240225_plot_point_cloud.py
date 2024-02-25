import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import multiprocessing as mp

from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.instruments.jwstnirspec_cal import filter_big_triangles

from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patheffects as PathEffects
import matplotlib.tri as tri

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass


    color_list = ["#006699","#ff9900", "#6600ff", "#006699", "#ff9900", "#6600ff"]

    ####################
    ## To be modified
    ####################
    fontsize = 12
    rad = 2 # arcsec
    # If True, make the size of the points proportional to the flux
    detector = "nrs2"
    # Wavelengths to plots
    wv0 = 4.5
    # Number of threads to be used for multithreading
    numthreads = 20
    # Number of nodes
    nodes = 40
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
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
    # init_centroid_nrs1 = [0.0,0.0]
    # init_centroid_nrs2 = [0.0,0.0]
    init_centroid_nrs1 = [-0.25973700664819993, 0.7535417070247359]
    init_centroid_nrs2 = [-0.2679950725373308, 0.7554649479920329]
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

    # List of preprocessing steps to be applied to each cal.fits file
    # It's possible to call each step directly as dataobj.compute_xyz(parameters,..) if you prefer.
    # This is only for convenience.
    preproc_task_list = []
    preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
    preproc_task_list.append(["compute_coordinates_arrays", {}, True, True])
    preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])  # old reduction, already in MJy
    preproc_task_list.append(["compute_quick_webbpsf_model", {"image_mask": None, "pixelscale": 0.1, "oversample": 10}, True, True])
    preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.0, "OWA": 0.5,"apply_offset":False,"init_centroid":init_centroid}, True, True])
    # preproc_task_list.append(["apply_coords_offset",{"coords_offset":(init_centroid[0],init_centroid[1])}])
    preproc_task_list.append(["compute_starspectrum_contnorm", {"mppool": mypool}, True, True])
    preproc_task_list.append(["compute_starsubtraction", {"mppool": mypool,"threshold_badpix": 50}, True, True])
    preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, True])

    # Make a list of all the data objects to be given to fit psf
    dataobj_list = []
    new_centroid_list = []
    for filename in nrs_filelist:
        print(filename)
        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
        dataobj_list.append(dataobj)

        new_centroid = dataobj.reload_new_coords_from_webbPSFfit(apply_offset=False)
        new_centroid_list.append(new_centroid)
        print("new_centroid",os.path.basename(filename),new_centroid)

    new_centroid_med = np.nanmedian(new_centroid_list,axis=0)
    print("Centroid from quick WebbPSF fit: (dra, ddec)=({0},{1}) in arcseconds".format(new_centroid_med[0],new_centroid_med[1]))

    dataobj0 = dataobj_list[0]

    reload_interpdata_outputs = dataobj0.reload_interpdata_regwvs(load_filename=dataobj0.default_filenames["compute_interpdata_regwvs"])
    all_interp_ra, all_interp_dec, all_interp_wvs, all_interp_flux, all_interp_err, all_interp_badpix, all_interp_area2d = reload_interpdata_outputs

    wv_sampling = dataobj0.wv_sampling
    if len(dataobj_list) > 1:
        for dataobj in dataobj_list[1::]:
            reload_interpdata_outputs = dataobj.reload_interpdata_regwvs(load_filename=dataobj.default_filenames["compute_interpdata_regwvs"])
            interp_ra, interp_dec, interp_wvs, interp_flux, interp_err, interp_badpix, interp_area2d = reload_interpdata_outputs

            all_interp_ra = np.concatenate((all_interp_ra, interp_ra), axis=0)
            all_interp_dec = np.concatenate((all_interp_dec, interp_dec), axis=0)
            all_interp_wvs = np.concatenate((all_interp_wvs, interp_wvs), axis=0)
            all_interp_flux = np.concatenate((all_interp_flux, interp_flux), axis=0)
            all_interp_err = np.concatenate((all_interp_err, interp_err), axis=0)
            all_interp_badpix = np.concatenate((all_interp_badpix, interp_badpix), axis=0)
            all_interp_area2d = np.concatenate((all_interp_area2d, interp_area2d), axis=0)

    mypool.close()
    mypool.join()

    l0 = np.argmin(np.abs(wv_sampling - wv0))

    plt.subplot(1, 3, 1)
    plt.scatter(all_interp_ra[:, l0], all_interp_dec[:, l0], s=1, c=color_list[0])
    txt = plt.text(0.03, 0.99, "Point cloud - $\lambda$={0} $\mu$m".format(wv0), fontsize=fontsize, ha='left', va='top',
                   transform=plt.gca().transAxes, color="black")
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for x in np.arange(-rad, rad, 0.1):
        plt.plot([x, x], [-rad, rad], color="grey", linewidth=1, alpha=0.5)
        plt.plot([-rad, rad], [x, x], color="grey", linewidth=1, alpha=0.5)
    plt.xlim([-rad, rad])
    plt.ylim([-rad, rad])
    plt.gca().set_aspect('equal')
    plt.xlabel("$\Delta$RA (as)", fontsize=fontsize)
    plt.ylabel("$\Delta$Dec (as)", fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.gca().invert_xaxis()

    plt.subplot(1, 3, 2)
    plt.scatter(all_interp_ra[:, l0], all_interp_dec[:, l0]- new_centroid_med[1],
                s=10 * all_interp_flux[:, l0] / np.nanmax(all_interp_flux[:, l0]), c=color_list[0])
    txt = plt.text(0.03, 0.99, "Point cloud (flux scaled) - $\lambda$={0} $\mu$m".format(wv0), fontsize=fontsize, ha='left',
                   va='top', transform=plt.gca().transAxes, color="black")
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for x in np.arange(-rad, rad, 0.1):
        plt.plot([x, x], [-rad, rad], color="grey", linewidth=1, alpha=0.5)
        plt.plot([-rad, rad], [x, x], color="grey", linewidth=1, alpha=0.5)
    plt.xlim([-rad, rad])
    plt.ylim([-rad, rad])
    plt.gca().set_aspect('equal')
    plt.xlabel("$\Delta$RA (as)", fontsize=fontsize)
    plt.ylabel("$\Delta$Dec (as)", fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.gca().invert_xaxis()
    # plt.show()

    plt.subplot(1, 3, 3)
    where_good = np.where(np.isfinite(all_interp_badpix[:, l0]))
    X = all_interp_ra[where_good[0], l0]- new_centroid_med[0]
    Y = all_interp_dec[where_good[0], l0]- new_centroid_med[1]
    Z = all_interp_flux[where_good[0], l0]
    filtered_triangles = filter_big_triangles(X, Y, 0.2)
    # Create filtered triangulation
    filtered_tri = tri.Triangulation(X, Y, triangles=filtered_triangles)
    # Perform LinearTriInterpolator for filtered triangulation
    pointcloud_interp = tri.LinearTriInterpolator(filtered_tri, Z)

    dra = 0.01
    ddec = 0.01
    ra_vec = np.arange(-2.5, 2.1, dra)
    dec_vec = np.arange(-3.0, 1.9, ddec)
    ra_grid, dec_grid = np.meshgrid(ra_vec, dec_vec)
    r_grid = np.sqrt(ra_grid ** 2 + dec_grid ** 2)

    myextent = [ra_vec[0] - dra / 2., ra_vec[-1] + dra / 2., dec_vec[0] - ddec / 2., dec_vec[-1] + ddec / 2.]
    myinterpim = pointcloud_interp(ra_grid, dec_grid)
    plt.imshow(np.log10(myinterpim), interpolation="nearest", origin="lower", extent=myextent)
    # plt.clim([0,np.nanmax(myinterpim)/4.0])
    txt = plt.text(0.03, 0.99, "log10(inteprolated flux) \n Centroid corrected \n $\lambda$={0} $\mu$m".format(wv0), fontsize=fontsize, ha='left',
                   va='top', transform=plt.gca().transAxes, color="black")
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for x in np.arange(-rad, rad, 0.1):
        plt.plot([x, x], [-rad, rad], color="grey", linewidth=1, alpha=0.5)
        plt.plot([-rad, rad], [x, x], color="grey", linewidth=1, alpha=0.5)
    plt.xlim([-rad, rad])
    plt.ylim([-rad, rad])
    plt.gca().set_aspect('equal')
    plt.xlabel("$\Delta$RA (as)", fontsize=fontsize)
    plt.ylabel("$\Delta$Dec (as)", fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.gca().invert_xaxis()

    plt.tight_layout()


    plt.show()
