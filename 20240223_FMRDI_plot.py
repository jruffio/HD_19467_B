import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as fits
from glob import glob
import multiprocessing as mp
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const

from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal

from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass


    ####################
    ## To be modified
    ####################
    # Use RDI PSF model as prior if True
    FMRDI = True
    # Directory containing the generated cal.fits file containing the best-fit PSF model from RDI
    RDImodel_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils/RDI_model_refPSF1"
    # Number of threads to be used for multithreading
    numthreads = 20
    # Number of nodes
    nodes = 40
    # Number of principal components (Karhunen-Loeve) modes to be added to the forward model
    N_KL = 3  # 0#3
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir = "/stow/jruffio/data/JWST/crds_cache/"
    # External_dir should external files like the NIRCam filters
    external_dir = "/stow/jruffio/data/JWST/external/"
    # Science data: List of stage 2 cal.fits files
    filelist = glob(
        "/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
    # utility folder where the intermediate and final data product will be saved
    utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils_fm/"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    # spectrum to be used for the companion template
    RDI_spec_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/HD19467b_RDI_1dspectrum_MJy.fits"
    ####################
    ## No need to change for HD19467B prog. id 1414
    # NIrcam filter used for flux normalization
    photfilter_name_nrs1 = "F360M"
    photfilter_name_nrs2 = "F460M"
    # Definition of the wavelength sampling on which the detector images are interpolated (for each detector)
    wv_sampling_nrs1 = np.arange(2.859509, 4.1012874, 0.0006763935)
    wv_sampling_nrs2 = np.arange(4.081285, 5.278689, 0.0006656647)
    # definiton of the nodes for the spline
    x_nodes_nrs1 = np.linspace(2.859509, 4.1012874, nodes, endpoint=True)
    x_nodes_nrs2 = np.linspace(4.081285, 5.278689, nodes, endpoint=True)
    # if 1: # Different attempt at define the nodes
    #     dw1,dw2 = 0.02,0.04
    #     l0,l2 = 2.859509-dw1, 4.1012874
    #     l1 = (l0+l2)/2
    #     x1_nodes = np.arange(l0,l1, dw1)
    #     l1 = x1_nodes[-1]+dw2
    #     x2_nodes = np.arange(l1,l2, dw2)
    #     x_nodes_nrs1 = np.concatenate([x1_nodes,x2_nodes])
    # if 1:
    #     dw1,dw2 = 0.04,0.02
    #     l0,l2 = 4.081285,5.278689+dw2
    #     l1 = (l0+l2)/2
    #     x1_nodes = np.arange(l0,l1, dw1)
    #     l1 = x1_nodes[-1]+dw2
    #     x2_nodes = np.arange(l1,l2, dw2)
    #     x_nodes_nrs2 = np.concatenate([x1_nodes,x2_nodes])
    # List of empirical offsets to correct the wcs coordinate system
    centroid_nrs1 = [-0.13499443, -0.07202978]
    centroid_nrs2 = [-0.12986898, -0.08441719]
    # RA Dec offset of the companion HD19467B
    ra_offset = -1332.871 / 1000.  # ra offset in as
    dec_offset = -875.528 / 1000.  # dec offset in as
    # Offsets for HD 19467 B from https://www.whereistheplanet.com/
    # RA Offset = -1332.871 +/- 10.886 mas
    # Dec Offset = -875.528 +/- 12.360 mas
    # Separation = 1593.703 +/- 9.530 mas
    # PA = 236.712 +/- 0.483 deg
    # Reference: Brandt et al. 2021
    # Absolute fluxes for the host star to be used in calculated flux ratios with the companion.
    HD19467_flux_MJy = {"F250M": 3.51e-6,  # in MJy, Ref Greenbaum+2023
                        "F300M": 2.63e-6,
                        "F335M": 2.10e-6,
                        "F360M": 1.82e-6,
                        "F410M": 1.49e-6,
                        "F430M": 1.36e-6,
                        "F460M": 1.12e-6}
    # Flux Calibration parameters
    flux_calib_paras = [-0.03864459, 1.09360589]
    ####################

    mypool = mp.Pool(processes=numthreads)

    for filename in filelist:
        print(filename)

        if "nrs1" in filename:
            continue
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

        try:
            photfilter = os.path.join(external_dir, "JWST_NIRCam." + photfilter_name + ".dat")
            filter_arr = np.loadtxt(photfilter)
        except:
            photfilter = os.path.join(external_dir, "Keck_NIRC2." + photfilter_name + ".dat")
            filter_arr = np.loadtxt(photfilter)
        trans_wvs = filter_arr[:, 0] / 1e4
        trans = filter_arr[:, 1]
        photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
        photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
        bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
        photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]
        print(photfilter_wvmin, photfilter_wvmax)

        preproc_task_list = []
        preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
        preproc_task_list.append(["compute_coordinates_arrays"])
        preproc_task_list.append(["convert_MJy_per_sr_to_MJy"])
        preproc_task_list.append(["apply_coords_offset", {"coords_offset": centroid}])
        preproc_task_list.append(
            ["compute_quick_webbpsf_model", {"image_mask": None, "pixelscale": 0.1, "oversample": 10}, True, True])
        # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
        preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}])
        preproc_task_list.append(
            ["compute_starspectrum_contnorm", {"x_nodes": x_nodes, "mppool": mypool}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"mppool": mypool}, True, True])
        # preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, True])

        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True, load_utils=True, preproc_task_list=preproc_task_list)
        # plt.imshow(dataobj.area2d,origin="lower")
        # plt.colorbar()
        # plt.show()
        from breads.instruments.jwstnirspec_cal import where_point_source

        where2mask = where_point_source(dataobj, (ra_offset, dec_offset), 0.4)
        tmp_badpixels = copy(dataobj.bad_pixels)
        tmp_badpixels[where2mask] = np.nan
        subtracted_im, star_model, spline_paras, _ = dataobj.reload_starsubtraction()

        if 1:
            from breads.instruments.jwstnirspec_cal import PCA_wvs_axis

            ny, nx = dataobj.data.shape
            first_half = np.where(dataobj.wavelengths < wv_sampling[np.size(wv_sampling) // 2])
            second_half = np.where(dataobj.wavelengths > wv_sampling[np.size(wv_sampling) // 2])
            wv4pca, im4pcs, n4pca, bp4pca = copy(dataobj.wavelengths), copy(subtracted_im), copy(
                dataobj.noise), copy(tmp_badpixels)
            # wv4pca[second_half] = np.nan
            # im4pcs[second_half] = np.nan
            # n4pca[second_half] = np.nan
            bp4pca[second_half] = np.nan
            KLs_wvs_left, KLs_left = PCA_wvs_axis(wv4pca, im4pcs, n4pca, bp4pca,
                                                  np.nanmedian(dataobj.wavelengths) / (4 * dataobj.R), N_KL=N_KL)
            wv4pca, im4pcs, n4pca, bp4pca = copy(dataobj.wavelengths), copy(subtracted_im), copy(
                dataobj.noise), copy(tmp_badpixels)
            # wv4pca[first_half] = np.nan
            # im4pcs[first_half] = np.nan
            # n4pca[first_half] = np.nan
            bp4pca[first_half] = np.nan
            KLs_wvs_right, KLs_right = PCA_wvs_axis(wv4pca, im4pcs, n4pca, bp4pca,
                                                    np.nanmedian(dataobj.wavelengths) / (4 * dataobj.R), N_KL=N_KL)
            wv4pca, im4pcs, n4pca, bp4pca = copy(dataobj.wavelengths), copy(subtracted_im), copy(
                dataobj.noise), copy(tmp_badpixels)
            KLs_wvs_all, KLs_all = PCA_wvs_axis(wv4pca, im4pcs, n4pca, bp4pca,
                                                np.nanmedian(dataobj.wavelengths) / (4 * dataobj.R), N_KL=N_KL)
            wvs_KLs_f_list = []
            for k in range(KLs_left.shape[1]):
                KL_f = interp1d(KLs_wvs_left, KLs_left[:, k], bounds_error=False, fill_value=0.0, kind="cubic")
                wvs_KLs_f_list.append(KL_f)
            for k in range(KLs_right.shape[1]):
                KL_f = interp1d(KLs_wvs_right, KLs_right[:, k], bounds_error=False, fill_value=0.0, kind="cubic")
                wvs_KLs_f_list.append(KL_f)
        else:
            wvs_KLs_f_list = None
        # plt.figure(2)
        # for k in range(KLs_left.shape[1]):
        #     plt.subplot(KLs_left.shape[1],1,k+1)
        #     plt.plot(KLs_wvs_left,KLs_left[:,k],label="{0}".format(k))
        # plt.figure(3)
        # for k in range(KLs_right.shape[1]):
        #     plt.subplot(KLs_right.shape[1],1,k+1)
        #     plt.plot(KLs_wvs_right,KLs_right[:,k],label="{0}".format(k))
        # plt.figure(4)
        # for k in range(KLs_all.shape[1]):
        #     plt.subplot(KLs_all.shape[1],1,k+1)
        #     plt.plot(KLs_wvs_all,KLs_all[:,k],label="{0}".format(k))
        # # plt.legend()
        # plt.show()

        if 1:  # read and normalize model grid
            RDI_spec_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/20230705_HD19467b_RDI_1dspectrum.fits"
            hdulist_sc = fits.open(RDI_spec_filename)
            grid_wvs = hdulist_sc[0].data
            grid_specs = ((hdulist_sc[1].data[None, None, :] * u.MJy) * (const.c / (grid_wvs * u.um) ** 2)).to(
                u.W * u.m ** -2 / u.um).value
            err = hdulist_sc[2].data

            grid_dwvs = grid_wvs[1::] - grid_wvs[0:np.size(grid_wvs) - 1]
            grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
            filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs))
            Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs)[None, None, :] * (
                        grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
            Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
            grid_specs = grid_specs / Fnu[:, :, None].to(u.MJy).value
            grid_specs = np.tile(grid_specs, (2, 2, 1))
            # grid_specs[np.where(np.isnan(grid_specs))] = 0
            myinterpgrid = RegularGridInterpolator(([-1, 1], [-1, 1]), grid_specs, method="linear",
                                                   bounds_error=False, fill_value=np.nan)
            teff, logg, vsini, rv, dra_comp, ddec_comp = 0.0, 0.0, 0.0, None, None, None
            fix_parameters = [teff, logg, vsini, rv, dra_comp, ddec_comp]

        # Definition of the priors
        if 1:
            subtracted_im, star_model, spline_paras, x_nodes = dataobj.reload_starsubtraction()

            wherenan = np.where(np.isnan(spline_paras))
            reg_mean_map = copy(spline_paras)
            reg_mean_map[wherenan] = \
            np.tile(np.nanmedian(spline_paras, axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
            if 0:
                reg_std_map = np.zeros(spline_paras.shape) + 300e-12  # np.nanmedian(spline_paras)
            else:
                reg_std_map = np.abs(spline_paras)
                reg_std_map[wherenan] = \
                np.tile(np.nanmax(np.abs(spline_paras), axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
                # reg_std_map[wherenan] = np.tile(np.zeros(spline_paras.shape), (1, spline_paras.shape[1]))[wherenan]
                reg_std_map = reg_std_map
                reg_std_map = np.clip(reg_std_map, 1e-11, np.inf)  # in MJy
        if FMRDI:
            RDImodel_filename = os.path.join(RDImodel_dir, os.path.basename(filename))
            hdulist_sc = fits.open(RDImodel_filename)
            RDImodel_im_MJy_per_sr = hdulist_sc["SCI"].data
            RDImodel_im = dataobj.convert_MJy_per_sr_to_MJy(data_in_MJy_per_sr=RDImodel_im_MJy_per_sr)

            # plt.figure(1)
            # plt.subplot(1,3,1)
            # # aspect = 1950. / (colmax - colmin + 1)
            # plt.imshow(dataobj.data,origin="lower", aspect='auto')
            # plt.ylim([0,1000])
            # plt.clim([-1e-10,1e-10])
            # plt.subplot(1,3,2)
            # plt.imshow(RDImodel_im,origin="lower", aspect='auto')
            # plt.ylim([0,1000])
            # plt.clim([-1e-10,1e-10])
            # plt.subplot(1,3,3)
            # plt.imshow(dataobj.data-RDImodel_im,origin="lower", aspect='auto')
            # plt.ylim([0,1000])
            # plt.clim([-1e-10,1e-10])
            #
            dataobj.bad_pixels[np.where(RDImodel_im == 0)] = np.nan
            from breads.instruments.jwstnirspec_cal import normalize_rows

            threshold_badpix = 10
            out_normalize_rows = normalize_rows(RDImodel_im, dataobj.wavelengths, noise=dataobj.noise,
                                                badpixs=dataobj.bad_pixels,
                                                nodes=nodes,
                                                star_model=dataobj.star_func(dataobj.wavelengths),
                                                threshold=threshold_badpix, use_set_nans=False,
                                                mypool=mypool, regularization=True, reg_mean_map=reg_mean_map,
                                                reg_std_map=reg_std_map)
            DImodel_star_model, _, new_badpixs, subtracted_im, spline_paras0 = out_normalize_rows
            out_normalize_rows = normalize_rows(RDImodel_im, dataobj.wavelengths, noise=dataobj.noise,
                                                badpixs=new_badpixs,
                                                nodes=nodes,
                                                star_model=dataobj.star_func(dataobj.wavelengths),
                                                threshold=threshold_badpix, use_set_nans=False,
                                                mypool=mypool, regularization=True, reg_mean_map=spline_paras0,
                                                reg_std_map=spline_paras0)
            DImodel_star_model, _, new_badpixs, subtracted_im, spline_paras = out_normalize_rows

            # # This code can plot the histogram if the relative error of the RDI model
            # # This is how you can handwave the justification for using "reg_std_map = reg_mean_map / 5" below
            # plt.figure(2)
            # relerr_map =(dataobj.data*dataobj.bad_pixels-RDImodel_im*new_badpixs)/np.abs(dataobj.data)
            # hist, bins = np.histogram(relerr_map[np.where(np.isfinite(relerr_map))], bins=20*3, range=(-1, 1))
            # bin_centers = (bins[1::]+bins[0:np.size(bins)-1])/2.
            # plt.plot(bin_centers,hist/(np.nansum(hist)*(bins[1]-bins[0])),label="relerr_map")
            # plt.yscale("log")
            # plt.show()

            wherenan = np.where(np.isnan(spline_paras))
            reg_mean_map = copy(spline_paras)
            reg_mean_map[wherenan] = \
            np.tile(np.nanmedian(spline_paras, axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
            reg_std_map = reg_mean_map / 5
            # reg_std_map = reg_mean_map
            reg_std_map = np.clip(reg_std_map, 3e-12, np.inf)  # in MJy

        from breads.fm.hc_atmgrid_splinefm_jwst_nirspec_cal import hc_atmgrid_splinefm_jwst_nirspec_cal

        fm_paras = {"atm_grid": myinterpgrid, "atm_grid_wvs": grid_wvs, "star_func": dataobj.star_func,
                    "radius_as": 0.1, "badpixfraction": 0.75, "nodes": x_nodes, "fix_parameters": fix_parameters,
                    "wvs_KLs_f": wvs_KLs_f_list, "regularization": "user", "reg_mean_map": reg_mean_map,
                    "reg_std_map": reg_std_map}
        fm_func = hc_atmgrid_splinefm_jwst_nirspec_cal

        # /!\ Optional but recommended
        # Test the forward model for a fixed value of the non linear parameter.
        # Make sure it does not crash and look the way you want
        if 1:
            out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
            #ra_offset+dataobj.wpsf_ra_offset,dec_offset+dataobj.wpsf_dec_offset
            color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
            dist2comp_as = np.sqrt((dataobj.dra_as_array-ra_offset) ** 2 + (dataobj.ddec_as_array-dec_offset) ** 2)
            dist2host_as = np.sqrt((dataobj.dra_as_array) ** 2 + (dataobj.ddec_as_array) ** 2)
            mask_comp  = dist2comp_as<0.1
            mask_host  = dist2host_as<0.1
            mask = np.zeros(dataobj.data.shape)+np.nan
            mask[np.isfinite(dataobj.dra_as_array)] = 1

            fontsize = 12
            fig = plt.figure(1, figsize=(12, 6))
            gs = gridspec.GridSpec(3, 5, height_ratios=[0.0, 1,0.0], width_ratios=[0.1, 1,0.2,1,0.1])
            gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.0)
            # gs.update(left=0.00, right=1, bottom=0.0, top=1, wspace=0.0, hspace=0.0)

            colmin,colmax = 262,326
            rowcut = 315#318
            Nrows_max = 200
            N_reg = Nrows_max*nodes


            ax1 = plt.subplot(gs[1, 1])
            plt.imshow(dataobj.data,origin="lower",interpolation="nearest")
            plt.clim([0,1e-9])
            plt.text(0.03, 0.98, "NIRSpec Detector (NRS2)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
            plt.gca().annotate('Star', xy=(1700, 716), xytext=(1850, 716),color="black", fontsize=fontsize,
                arrowprops=dict(facecolor="black", shrink=0.02),horizontalalignment='left',verticalalignment='center')
            plt.gca().annotate('Area', xy=(1825, colmax+10), fontsize=fontsize,color="Red",horizontalalignment='center',verticalalignment='bottom')
            plt.gca().annotate('Companion', xy=(1825, colmin-10), fontsize=fontsize,color="Red",horizontalalignment='center',verticalalignment='top')
                # arrowprops=dict(facecolor=color_list[1], shrink=0.05))
            # contour = plt.gca().contour(mask.astype(int), colors="grey", levels=[0.5],alpha=1)
            # contour = plt.gca().contourf(mask_comp.astype(int), colors="#ff9900", levels=[0.5,1.5],alpha=1)
            # contour = plt.gca().contourf(mask_host.astype(int), colors="#006699", levels=[0.5,1.5],alpha=1)
            # plt.gca().set_aspect('equal')
            plt.xlim([0,2048])
            plt.ylim([0,2048])
            plt.fill_between([0,2048],[colmin-0.5,colmin-0.5],[colmax+0.5,colmax+0.5],color=color_list[0],alpha=0.5)
            plt.xlabel("Columns (pix)",fontsize=fontsize)
            plt.ylabel("Rows (pix)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)

            ax1 = plt.subplot(gs[1, 3])
            # contour = plt.gca().contour(mask.astype(int), colors="grey", levels=[0.5],alpha=1)
            plt.fill_between([0,2048],[colmin-0.5,colmin-0.5],[colmax+0.5,colmax+0.5],color=color_list[0],alpha=0.2, zorder=0)
            plt.imshow(dataobj.data,origin="lower",interpolation="nearest",extent=[-0.5,2048+0.5,-0.5,2048+0.5],aspect=1675./(colmax-colmin+1),alpha=0.5)
            plt.text(0.03, 0.98, "Zoom in companion area", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
            plt.clim([0.0e-10,1e-10])
            # contour = plt.gca().contourf(mask.astype(int), colors="grey", levels=[0.5,1.5],alpha=0.7)
            contour = plt.gca().contourf(mask_comp.astype(int), colors="red", levels=[0.5,1.5],alpha=1)
            # contour = plt.gca().contourf(mask_host.astype(int), colors="#006699", levels=[0.5,1.5],alpha=1)
            plt.gca().annotate('Companion Trace', xy=(800, colmax-10), xytext=(800, (colmin+colmax)/2-10),color="red", fontsize=fontsize,
                arrowprops=dict(facecolor=color_list[0], shrink=0.05),horizontalalignment='center',verticalalignment='center')
            plt.gca().annotate('', xy=(800, colmin+5), xytext=(800, (colmin+colmax)/2-12),color=color_list[0], fontsize=fontsize,
                arrowprops=dict(facecolor=color_list[0], shrink=0.05),horizontalalignment='center',verticalalignment='center')
            plt.xlim([0,1675])
            plt.ylim([colmin-0.5,colmax+0.5])
            plt.xlabel("Columns (pix)",fontsize=fontsize)
            plt.ylabel("Rows (pix)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)

            # out_filename = os.path.join(out_png, "FM_im.png")
            # print("Saving " + out_filename)
            # plt.savefig(out_filename, dpi=300)
            # plt.savefig(out_filename.replace(".png", ".pdf"))

            # plt.show()

            fig = plt.figure(2, figsize=(12, 8))
            gs = gridspec.GridSpec(6, 3, height_ratios=[0.00,0.2,0.2,0.2,0.2,0.2], width_ratios=[0.05, 1,0.2])
            gs.update(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.0, hspace=0.0)

            # fm_paras["fix_parameters"]= [None,None,None,sc_fib]
            print(ra_offset,dec_offset)
            nonlin_paras = [0.0,ra_offset,dec_offset]
            # d is the data vector a the specified location
            # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
            # s is the vector of uncertainties corresponding to d
            d, M, s,extra_outputs = fm_func(nonlin_paras,dataobj,return_extra_outputs=True,**fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            w = extra_outputs["wvs"]
            x = extra_outputs["ras"]
            y = extra_outputs["decs"]
            rows = extra_outputs["rows"]
            d_reg,s_reg = extra_outputs["regularization"]
            reg_wvs = extra_outputs["regularization_wvs"]
            reg_rows = extra_outputs["regularization_rows"]
            unique_rows = np.unique(rows)
            shifted_w = w+(wv_sampling[-1]-wv_sampling[0])*(rows-np.nanmin(unique_rows))
            shifted_reg_wvs = reg_wvs+(wv_sampling[-1]-wv_sampling[0])*(reg_rows-np.nanmin(unique_rows))



            validpara = np.where(np.max(np.abs(M),axis=0)!=0)
            M = M[:,validpara[0]]

            d = d / s
            M = M / s[:, None]

            from breads.fit import fitfm
            residuals = np.ones(np.size(s)+np.size(validpara[0]))+np.nan
            residuals_H0 = np.ones(np.size(s)+np.size(validpara[0]))+np.nan
            noise4residuals = np.ones(np.size(s)+np.size(validpara[0]))+np.nan
            log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,
                                                                         computeH0 = True,bounds = None,
                                                                         residuals=residuals,residuals_H0=residuals_H0,noise4residuals=noise4residuals,
                                                                         scale_noise=True,marginalize_noise_scaling=False)
            paras = linparas[validpara]
            print("best fit", linparas[0:5])
            print("best fit err",linparas_err[0:5])
            print("best fit snr",linparas[0:5]/linparas_err[0:5])
            print("rchi2",rchi2)

            logdet_Sigma = np.sum(2 * np.log(s))
            m = np.dot(M, paras)
            r = d  - m
            chi2 = np.nansum(r**2)
            N_data = np.size(d)
            rchi2 = chi2 / N_data
            res = r * s
            #
            for rowcutid,_rowcut in enumerate([313,315,317,319,321]):
                where_row = np.where(rows == _rowcut)
                ax1 = plt.subplot(gs[rowcutid+1, 1])
                plt.plot(w[where_row], (d * s)[where_row]*1e12, label="Data", color=color_list[1])
                plt.text(0.01, 0.98, "Row {0}".format(_rowcut), fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
                plt.fill_between(x_nodes,(reg_mean_map[_rowcut, :]-reg_std_map[_rowcut, :])*1e12,(reg_mean_map[_rowcut, :]+reg_std_map[_rowcut, :])*1e12,color="black",alpha=0.3)
                plt.plot(w[where_row], (m * s)[where_row]*1e12, label="Combined Model", color=color_list[0])
                plt.plot(w[where_row], (paras[0] * M[:, 0] * s)[where_row]*1e12, label="planet model", color="pink")
                plt.plot(w[where_row], ((m - paras[0] * M[:, 0]) * s)[where_row]*1e12, label="starlight model",color=color_list[2])
                plt.errorbar(x_nodes,reg_mean_map[_rowcut, :]*1e12,yerr=reg_std_map[_rowcut, :]*1e12,color="black", ls='none',label="RDI prior")
                if rowcutid == 0:
                    plt.legend(fontsize=fontsize, loc="upper left", bbox_to_anchor=(1.01, 1))
                    plt.ylim([-0.5e1, 20e1])
                plt.ylabel("Flux ($\mu$Jy)", fontsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                # plt.ylim([-0.5e-11, 15e-11])
                plt.xlim([w[where_row][0], w[where_row][-1]])
                if rowcutid != 4:
                    plt.gca().set_xticklabels([])
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)

            out_filename = os.path.join(out_png, "FMRDI_v2.png")
            print("Saving " + out_filename)
            plt.savefig(out_filename, dpi=300)
            plt.savefig(out_filename.replace(".png", ".pdf"))

            plt.show()

            exit()
            ## HEre stop


            M_full = copy(M/s[:,None])
            # s = np.ones(s.shape)
            # exit()
            mask_comp = np.zeros(dataobj.data.shape)
            mask_comp[where_finite] = 1
            with fits.open(dataobj.starsub_filename) as hdulist:
                subtracted_im = hdulist[0].data
            # plt.subplot(1,2,1)
            # plt.imshow(dataobj.wavelengths,origin="lower")
            # plt.subplot(1,2,2)
            # plt.imshow(subtracted_im,origin="lower")
            # contour = plt.gca().contour(mask_comp.astype(int), colors='cyan', levels=[0.5])
            # plt.clim([-1e-11,1e-11])
            # plt.show()

            # d,M,s = d[:1717],M[:1717,:],s[0:1717]
            # d,M,s = d[500:1000],M[500:1000,:],s[500:1000]

            # wvs_im = dataobj.wavelengths
            # cols_im = np.tile(np.arange(wvs_im.shape[1])[None,:],(wvs_im.shape[0],1))
            # d_wvs = wvs_im[where_finite]
            # d_cols = cols_im[where_finite]

            print(M.shape)
            # validpara = np.where(np.sum(M,axis=0)!=0)
            validpara = np.where(np.max(np.abs(M),axis=0)!=0)
            M = M[:,validpara[0]]
            # print(M.shape)
            # print(d.shape)
            # print(np.nanmax(M,axis=0))
            # print(np.sum(M,axis=0)!=0)
            # print(np.nanstd(M,axis=0))

            d = d / s
            M = M / s[:, None]
            from scipy.optimize import lsq_linear

            # plt.subplot(3,1,1)
            # plt.plot(d,label="data")
            # plt.subplot(3,1,2)
            # plt.plot(M[:,0]/np.max(M[:,0]),label="planet model")
            # # plt.legend()
            # plt.subplot(3,1,3)
            # for k in range(M.shape[-1]-1):
            #     plt.plot(M[:,k+1])
            # plt.legend()
            # plt.show()


        if 1:
            # (d_reg,s_reg) = extra_outputs["regularization"]
            # s_reg = s_reg[validpara]
            # d_reg = d_reg[validpara]
            # print(np.where(np.isnan(s_reg)))
            # where_reg = np.where(np.isfinite(s_reg))
            # s_reg = s_reg[where_reg]
            # d_reg = d_reg[where_reg]
            # M_reg = np.zeros((np.size(where_reg[0]),M.shape[1]))
            # M_reg[np.arange(np.size(where_reg[0])),where_reg[0]] = 1/s_reg
            #
            # M2 = np.concatenate([M,M_reg],axis=0)
            # d2 = np.concatenate([d,d_reg/s_reg])
            # paras = lsq_linear(M2,d2).x

            from breads.fit import fitfm
            log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0 = True,bounds = None,residuals=None,residuals_H0=None)
            # print("here")
            # linparas0 = np.zeros(M_full.shape[1])+np.nan
            # linparas0[validpara] = paras
            # print(linparas0)
            # print("fitfm")
            # print(linparas)
            paras = linparas[validpara]
            print("best fit", linparas[0:5])
            print("best fit err",linparas_err[0:5])
            print("best fit snr",linparas[0:5]/linparas_err[0:5])#[107.89260376   4.30915356   4.32027028   4.39081023   9.39263101]
            # exit()
        else:
            paras = lsq_linear(M, d).x
        # plt.figure(10)
        # plt.plot(paras)
        # plt.show()

        paras_H0 = lsq_linear(M[:,1::], d).x
        # m_H0 = np.dot(M[:,1::],paras_H0)
        print("best fit",paras[0:5])
        paras_full = np.zeros(M_full.shape[1])
        paras_full[validpara[0]] = paras

        # print((paras[0]*u.MJy *(const.c)/(photfilter_wv0*u.um)**2).to( u.W/u.m**2/u.um))
        # print((paras[0]*u.MJy *(const.c)/(photfilter_wv0*u.um)).to( u.W/u.m**2))

        logdet_Sigma = np.sum(2 * np.log(s))
        m = np.dot(M, paras)
        r = d  - m
        chi2 = np.nansum(r**2)
        N_data = np.size(d)
        rchi2 = chi2 / N_data
        # MTM = np.dot(M.T, M)
        # covphi = rchi2 * np.linalg.inv(MTM)
        # slogdet_icovphi0 = np.linalg.slogdet(MTM)

        # N_linpara = M.shape[1]
        # from scipy.special import loggamma
        # log_prob = -0.5 * logdet_Sigma - 0.5 * slogdet_icovphi0[1] - (N_data - N_linpara + 2 - 1) / 2 * np.log(chi2) + \
        #             loggamma((N_data - N_linpara + 2 - 1) / 2) + (N_linpara - N_data) / 2 * np.log(2 * np.pi)
        # diagcovphi = copy(np.diag(covphi))
        # diagcovphi[np.where(diagcovphi<0.0)] = np.nan
        # paras_err = np.sqrt(diagcovphi)
        # # print("best fit err",paras_err[0:5])
        # # print("SNR",paras[0]/paras_err[0])
        # # print("std(r)",np.nanstd(r))

        res = r * s

        for rowcutid,_rowcut in enumerate([313,315,317,319,321]):
            where_row = np.where(where_finite[0] == _rowcut)
            ax1 = plt.subplot(gs[rowcutid+1, 1])
            plt.plot(d_wvs[where_row], (d * s)[where_row]*1e12, label="Data", color=color_list[1])
            plt.text(0.01, 0.98, "Row {0}".format(_rowcut), fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="black")
            plt.fill_between(x_knots,(reg_mean_map[_rowcut, :]-reg_std_map[_rowcut, :])*1e12,(reg_mean_map[_rowcut, :]+reg_std_map[_rowcut, :])*1e12,color="black",alpha=0.3)
            plt.plot(d_wvs[where_row], (m * s)[where_row]*1e12, label="Combined Model", color=color_list[0])
            plt.plot(d_wvs[where_row], (paras[0] * M[:, 0] * s)[where_row]*1e12, label="planet model", color="pink")
            plt.plot(d_wvs[where_row], ((m - paras[0] * M[:, 0]) * s)[where_row]*1e12, label="starlight model",color=color_list[2])
            plt.errorbar(x_knots,reg_mean_map[_rowcut, :]*1e12,yerr=reg_std_map[_rowcut, :]*1e12,color="black", ls='none',label="RDI prior")
            if rowcutid == 0:
                plt.legend(fontsize=fontsize, loc="upper left", bbox_to_anchor=(1.01, 1))
                plt.ylim([-0.5e1, 20e1])
            plt.ylabel("Flux ($\mu$Jy)", fontsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            # plt.ylim([-0.5e-11, 15e-11])
            plt.xlim([d_wvs[where_row][0], d_wvs[where_row][-1]])
            if rowcutid != 4:
                plt.gca().set_xticklabels([])
        plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)

        out_filename = os.path.join(out_png, "FMRDI_v2.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        plt.show()

        if 0:
            where_row =np.where(where_finite[0]==rowcut)
            ax1 = plt.subplot(gs[1, 1])
            plt.plot(d_wvs[where_row],(d*s)[where_row],label="Data",color=color_list[1])
            plt.plot(d_wvs[where_row],(m*s)[where_row],label="Combined Model",color=color_list[0])
            plt.plot(d_wvs[where_row],(paras[0]*M[:,0]*s)[where_row],label="planet model",color="pink")
            plt.plot(d_wvs[where_row],((m-paras[0]*M[:,0])*s)[where_row],label="starlight model",color=color_list[2])
            plt.legend(fontsize=fontsize,loc="upper left", bbox_to_anchor=(1.01, 1))
            plt.ylabel("Flux (MJy)",fontsize=fontsize)
            plt.gca().set_xticklabels([])
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.ylim([-0.5e-11, 8e-11])
            plt.xlim([d_wvs[where_row][0],d_wvs[where_row][-1]])

            ax1 = plt.subplot(gs[2, 1])
            startid = np.where(np.unique(where_finite[0]) == rowcut)[0][0]*nodes+1
            # print(np.where(np.unique(where_finite[0]) == rowcut)[0][0], startid,nodes)
            mrow = np.dot(M_full[:, startid:startid+nodes], paras_full[startid:startid+nodes])
            # mrow2 = np.dot(M_full[:, startid+1:startid+nodes], paras_full[startid+1:startid+nodes])
            # mrow3 = np.dot(M_full[:, startid+2:startid+nodes], paras_full[startid+2:startid+nodes])
            # mrow4 = np.dot(M_full[:, startid+3:startid+nodes], paras_full[startid+3:startid+nodes])
            # mrow5 = np.dot(M_full[:, startid+4:startid+nodes], paras_full[startid+4:startid+nodes])
            plt.plot(d_wvs[where_row],(mrow*s)[where_row],label="starlight model",color=color_list[2])
            # plt.plot(d_wvs[where_row],(mrow2*s)[where_row],label="starlight model2")
            # plt.plot(d_wvs[where_row],(mrow3*s)[where_row],label="starlight model3")
            # plt.plot(d_wvs[where_row],(mrow4*s)[where_row],label="starlight model4")
            # plt.plot(d_wvs[where_row],(mrow5*s)[where_row],label="starlight model5")
            # plt.plot(d_wvs[where_row],(mrow*s)[where_row],label="test",color=color_list[1],linestyle="--")
            # plt.show()
            # for k in np.arange(1,Nrows_max*nodes+1):
            #     if np.nansum(M_full[where_row[0],k]!=0) ==0:
            #         continue
            for k in np.arange(startid,startid+nodes): #start 602
                # print("spline",k)
                # plt.plot(d_wvs[where_row],(mrow*s)[where_row],label="starlight model",color=color_list[2])
                if k !=startid+20:
                    plt.plot(d_wvs[where_row],((paras_full[k]*M_full[:,k])*s)[where_row],linestyle="--", color=color_list[2],alpha=0.5)
                else:
                    plt.plot(d_wvs[where_row],((paras_full[k]*M_full[:,k])*s)[where_row],linestyle="--", color=color_list[2],alpha=0.5,label="Sub-components")
            plt.errorbar(x_knots,reg_mean_map[rowcut, :],yerr=reg_std_map[rowcut, :],color="black", ls='none',label="RDI prior")
            plt.legend(fontsize=fontsize,loc="upper left", bbox_to_anchor=(1.01, 1))
            plt.ylabel("Flux (MJy)",fontsize=fontsize)
            plt.gca().set_xticklabels([])
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.ylim([-0.5e-11, 8e-11])
            plt.xlim([d_wvs[where_row][0],d_wvs[where_row][-1]])

            ax1 = plt.subplot(gs[3, 1])
            startid = Nrows_max*nodes+1+np.where(np.unique(where_finite[0]) == rowcut)[0][0]*len(wvs_KLs_f_list)
            print(startid)
            # for k in np.arange(Nrows_max*nodes+1,Nrows_max*(nodes+len(wvs_KLs_f_list))+1):
            #     if np.nansum(M_full[where_row[0],k]!=0) ==0:
            #         continue
            for k in np.arange(startid,startid+len(wvs_KLs_f_list)//2): #8091
                # print("kls",k)
                if k !=startid:
                    plt.plot(d_wvs[where_row],((paras_full[k]*M_full[:,k])*s)[where_row],linestyle="-", color="grey",alpha=0.5)
                else:
                    plt.plot(d_wvs[where_row],((paras_full[k]*M_full[:,k])*s)[where_row],linestyle="-", color="grey",alpha=0.5,label="Left PCAs")
            plt.legend(fontsize=fontsize,loc="upper left", bbox_to_anchor=(1.01, 1))
            plt.gca().set_xticklabels([])
            plt.ylabel("Flux (MJy)",fontsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.gca().set_yticks([-2e-13,0,2e-13])
            plt.ylim([-4e-13, 4e-13])
            plt.xlim([d_wvs[where_row][0],d_wvs[where_row][-1]])

            ax1 = plt.subplot(gs[4, 1])
            for k in np.arange(startid+len(wvs_KLs_f_list)//2,startid+len(wvs_KLs_f_list)): #8091+3
                print("kls",k)
                if k !=startid+len(wvs_KLs_f_list)//2:
                    plt.plot(d_wvs[where_row],((paras_full[k]*M_full[:,k])*s)[where_row],linestyle="-", color="grey",alpha=0.5)
                else:
                    plt.plot(d_wvs[where_row],((paras_full[k]*M_full[:,k])*s)[where_row],linestyle="-", color="grey",alpha=0.5,label="Right PCAs")
            plt.legend(fontsize=fontsize,loc="upper left", bbox_to_anchor=(1.01, 1))
            plt.gca().set_xticklabels([])
            plt.ylabel("Flux (MJy)",fontsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.gca().set_yticks([-2e-13,0,2e-13])
            plt.ylim([-4e-13, 4e-13])
            plt.xlim([d_wvs[where_row][0],d_wvs[where_row][-1]])


            ax1 = plt.subplot(gs[5, 1])
            plt.scatter(d_wvs[where_row],(res)[where_row],label="Residuals",color="black",s=1)
            plt.fill_between(d_wvs[where_row],-(s)[where_row],(s)[where_row],label="Flux error",alpha=0.5,color="grey")
            plt.legend(fontsize=fontsize,loc="upper left", bbox_to_anchor=(1.01, 1))
            plt.gca().set_yticks([-1e-11,0,1e-11])
            plt.ylim([-1.5e-11, 1.5e-11])
            plt.xlim([d_wvs[where_row][0],d_wvs[where_row][-1]])
            plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
            plt.ylabel("Flux (MJy)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)

            out_filename = os.path.join(out_png, "FMRDI.png")
            print("Saving " + out_filename)
            plt.savefig(out_filename, dpi=300)
            plt.savefig(out_filename.replace(".png", ".pdf"))

            plt.show()
            exit()
