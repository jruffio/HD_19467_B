import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as fits
from glob import glob
import multiprocessing as mp
import h5py
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const
from scipy.signal import correlate2d

from breads.instruments.jwstnirspec_cal import JWSTNirspec_cal
from breads.grid_search import grid_search
from breads.fm.hc_atmgrid_splinefm_jwst_nirspec_cal import hc_atmgrid_splinefm_jwst_nirspec_cal
from breads.instruments.jwstnirspec_cal import PCA_wvs_axis
from breads.instruments.jwstnirspec_cal import where_point_source
from breads.fit import fitfm

from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')

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
    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
    # utility folder where the intermediate and final data product will be saved
    utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_utils_fm/"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    # out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_out_fm/xy/"
    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240216_out_fm/xy/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # spectrum to be used for the companion template
    RDI_spec_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/HD19467b_RDI_1dspectrum_MJy.fits"
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
    ra_offset = -1332.871/1000. # ra offset in as
    dec_offset = -875.528/1000. # dec offset in as
        # Offsets for HD 19467 B from https://www.whereistheplanet.com/
        # RA Offset = -1332.871 +/- 10.886 mas
        # Dec Offset = -875.528 +/- 12.360 mas
        # Separation = 1593.703 +/- 9.530 mas
        # PA = 236.712 +/- 0.483 deg
        # Reference: Brandt et al. 2021
    # Absolute fluxes for the host star to be used in calculated flux ratios with the companion.
    HD19467_flux_MJy = {"F250M":3.51e-6, # in MJy, Ref Greenbaum+2023
                         "F300M":2.63e-6,
                         "F335M":2.10e-6,
                         "F360M":1.82e-6,
                         "F410M":1.49e-6,
                         "F430M":1.36e-6,
                         "F460M":1.12e-6}
    # Flux Calibration parameters
    flux_calib_paras = [-0.03864459,  1.09360589]
    ####################


    mypool = mp.Pool(processes=numthreads)

    for filename in filelist[0::]:
        print(filename)

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
        preproc_task_list.append(["apply_coords_offset",{"coords_offset": centroid}])
        preproc_task_list.append(["compute_quick_webbpsf_model", {"image_mask": None, "pixelscale": 0.1, "oversample": 10}, True, True])
        # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
        preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}])
        preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": x_nodes, "mppool": mypool}, True, True])
        preproc_task_list.append(["compute_starsubtraction", {"mppool": mypool}, True, True])
        # preproc_task_list.append(["compute_interpdata_regwvs", {"wv_sampling": wv_sampling}, True, True])

        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)
        # plt.imshow(dataobj.area2d,origin="lower")
        # plt.colorbar()
        # plt.show()
    #     plt.plot(wv_sampling,dataobj.star_func(wv_sampling))
    # plt.show()
    # if 1:
        where2mask = where_point_source(dataobj,(ra_offset,dec_offset),0.4)
        tmp_badpixels = copy(dataobj.bad_pixels)
        tmp_badpixels[where2mask] = np.nan
        subtracted_im,star_model,spline_paras,_ = dataobj.reload_starsubtraction()

        if 1:
            ny,nx = dataobj.data.shape
            first_half = np.where(dataobj.wavelengths<wv_sampling[np.size(wv_sampling)//2])
            second_half = np.where(dataobj.wavelengths>wv_sampling[np.size(wv_sampling)//2])
            wv4pca,im4pcs,n4pca,bp4pca =  copy(dataobj.wavelengths),copy(subtracted_im),copy(dataobj.noise),copy(tmp_badpixels)
            # wv4pca[second_half] = np.nan
            # im4pcs[second_half] = np.nan
            # n4pca[second_half] = np.nan
            bp4pca[second_half] = np.nan
            KLs_wvs_left,KLs_left = PCA_wvs_axis(wv4pca,im4pcs,n4pca,bp4pca,
                               np.nanmedian(dataobj.wavelengths)/(4*dataobj.R),N_KL=N_KL)
            wv4pca,im4pcs,n4pca,bp4pca =  copy(dataobj.wavelengths),copy(subtracted_im),copy(dataobj.noise),copy(tmp_badpixels)
            # wv4pca[first_half] = np.nan
            # im4pcs[first_half] = np.nan
            # n4pca[first_half] = np.nan
            bp4pca[first_half] = np.nan
            KLs_wvs_right,KLs_right = PCA_wvs_axis(wv4pca,im4pcs,n4pca,bp4pca,
                               np.nanmedian(dataobj.wavelengths)/(4*dataobj.R),N_KL=N_KL)
            wv4pca,im4pcs,n4pca,bp4pca =  copy(dataobj.wavelengths),copy(subtracted_im),copy(dataobj.noise),copy(tmp_badpixels)
            KLs_wvs_all,KLs_all = PCA_wvs_axis(wv4pca,im4pcs,n4pca,bp4pca,
                               np.nanmedian(dataobj.wavelengths)/(4*dataobj.R),N_KL=N_KL)
            wvs_KLs_f_list = []
            for k in range(KLs_left.shape[1]):
                KL_f = interp1d(KLs_wvs_left,KLs_left[:,k], bounds_error=False, fill_value=0.0,kind="cubic")
                wvs_KLs_f_list.append(KL_f)
            for k in range(KLs_right.shape[1]):
                KL_f = interp1d(KLs_wvs_right,KLs_right[:,k], bounds_error=False, fill_value=0.0,kind="cubic")
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

        if 1: # read and normalize companion template used in FM
            hdulist_sc = fits.open(RDI_spec_filename)
            grid_wvs = hdulist_sc[0].data
            RDI_spec = ((hdulist_sc[1].data*u.MJy)*(const.c/(grid_wvs*u.um)**2)).to(u.W*u.m**-2/u.um).value
            err = hdulist_sc[2].data

            # add flux calibration correction because we are fitting a pure WebbPSF model later, which comes with systematics
            RDI_spec = RDI_spec/np.polyval(flux_calib_paras, grid_wvs)

            grid_dwvs = grid_wvs[1::]-grid_wvs[0:np.size(grid_wvs)-1]
            grid_dwvs = np.insert(grid_dwvs,0,grid_dwvs[0])
            filter_norm = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs))
            Flambda = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs)*(RDI_spec*u.W*u.m**-2/u.um))/filter_norm
            Fnu = Flambda*(photfilter_wv0*u.um)**2/const.c # from Flambda back to Fnu
            RDI_spec = RDI_spec/Fnu.to(u.MJy).value

            # The code expects a model grid but we have a single spectrum, so we are creating a dumb grid but fixing the parameters anyway...
            grid_specs = np.tile(RDI_spec[None,None,:],(2,2,1))
            # grid_specs[np.where(np.isnan(grid_specs))] = 0
            myinterpgrid = RegularGridInterpolator(([-1,1],[-1,1]),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
            teff,logg,vsini,rv,dra_comp,ddec_comp = 0.0,0.0,0.0,None,None,None
            fix_parameters = [teff,logg,vsini,rv,dra_comp,ddec_comp]

        if 1: # Definition of the priors
            subtracted_im,star_model,spline_paras,x_nodes = dataobj.reload_starsubtraction()

            wherenan = np.where(np.isnan(spline_paras))
            reg_mean_map = copy(spline_paras)
            reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras, axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
            if 0:
                reg_std_map = np.zeros(spline_paras.shape)+300e-12#np.nanmedian(spline_paras)
            else:
                reg_std_map = np.abs(spline_paras)
                reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras), axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
                # reg_std_map[wherenan] = np.tile(np.zeros(spline_paras.shape), (1, spline_paras.shape[1]))[wherenan]
                reg_std_map = reg_std_map
                reg_std_map = np.clip(reg_std_map, 1e-11, np.inf) # in MJy

        fm_paras = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"star_func":dataobj.star_func,
                    "radius_as":0.1,"badpixfraction":0.75,"nodes":x_nodes,"fix_parameters":fix_parameters,
                    "wvs_KLs_f":wvs_KLs_f_list,"regularization":"user","reg_mean_map":reg_mean_map,"reg_std_map":reg_std_map}
        fm_func = hc_atmgrid_splinefm_jwst_nirspec_cal

        # /!\ Optional but recommended
        # Test the forward model for a fixed value of the non linear parameter.
        # Make sure it does not crash and look the way you want
        if 0:
            # fm_paras["fix_parameters"]= [None,None,None,sc_fib]
            print(ra_offset,dec_offset)
            nonlin_paras = [0.0,ra_offset,dec_offset] # rv (km/s), dra (arcsec),ddec (arcsec),
            # nonlin_paras = [0, -0.9,-0.25] # rv (km/s), dra (arcsec),ddec (arcsec),
            # nonlin_paras = [0, -1.4, -1.4] # rv (km/s), dra (arcsec),ddec (arcsec),
            # nonlin_paras = [0, 0.5, -0.5]
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

            # print(np.unique(where_finite[0]))
            # mask_comp = np.zeros(dataobj.data.shape)
            # mask_comp[where_finite] = 1


            validpara = np.where(np.max(np.abs(M),axis=0)!=0)
            M = M[:,validpara[0]]

            d = d / s
            M = M / s[:, None]

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
            plt.figure(10)
            plt.plot(linparas,label="linparas")
            plt.plot(linparas_err,label="linparas_err")
            plt.legend()


            logdet_Sigma = np.sum(2 * np.log(s))
            m = np.dot(M, paras)
            r = d  - m
            chi2 = np.nansum(r**2)
            N_data = np.size(d)
            rchi2 = chi2 / N_data
            res = r * s

            plt.figure(1)
            plt.subplot(3,1,1)
            plt.plot(shifted_w,d*s,label="data")
            plt.plot(shifted_w,m*s,label="Combined model")
            plt.plot(shifted_w,paras[0]*M[:,0]*s,label="planet model")
            plt.plot(shifted_w,(m-paras[0]*M[:,0])*s,label="starlight model")
            where_even_rows = np.where((reg_rows % 2)==0)
            plt.errorbar(shifted_reg_wvs[where_even_rows],d_reg[where_even_rows],yerr=s_reg[where_even_rows],label="even rows prior")
            where_odd_rows = np.where((reg_rows % 2)==1)
            plt.errorbar(shifted_reg_wvs[where_odd_rows],d_reg[where_odd_rows],yerr=s_reg[where_odd_rows],label="odd rows prior")
            plt.ylabel("Flux (MJy)")
            plt.xlabel("Column pixels")
            plt.legend()

            plt.subplot(3,1,2)
            plt.plot(shifted_w,r,label="Residuals")
            # plt.fill_between(np.arange(np.size(s)),-1,1,label="Error",alpha=0.5)
            r_std = np.nanstd(r)
            plt.ylim([-10*r_std,10*r_std])
            plt.ylabel("Flux (MJy)")
            plt.xlabel("Column pixels")
            plt.legend()

            plt.subplot(3,1,3)
            plt.plot(shifted_w,M[:,0],label="planet model")

            # plt.subplot(4,1,4)
            # for k in range(np.min([50,M.shape[-1]-1])):
            #     plt.plot(shifted_w,M[:,k+1],label="starlight model {0}".format(k+1))
            # # plt.legend()

            plt.figure(2)
            plt.subplot(3,1,1)
            plt.scatter(w,res,alpha=0.5,s=0.1)
            plt.xlabel("Wavelength (um)")
            plt.subplot(3,1,2)
            plt.scatter(rows,res,alpha=0.5,s=0.1)
            plt.xlabel("Column index")
            plt.tight_layout()
            plt.subplot(3,1,3)
            fcal= "/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/stage2_notelegraph/jw01414004001_02101_00003_nrs1_cal.fits"
            hdulist_sc = fits.open(fcal)
            im = hdulist_sc["SCI"].data
            # plt.plot(im[220,:])
            random_ints = np.array([219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 268, 269])
            for row in random_ints:#random_ints:
                plt.plot(im[row,:],alpha=0.5)
            plt.ylim([-1e-10,1e-10])

            plt.figure(3)
            cumchi2_H1 = np.nancumsum(residuals**2)
            cumchi2_H0 = np.nancumsum(residuals_H0**2)
            plt.subplot(1,2,1)
            plt.plot(shifted_w,(cumchi2_H1-cumchi2_H0)[0:np.size(shifted_w)],linestyle="--",label="cumchi2_H1-cumchi2_H0 (cropped)")
            plt.subplot(1,2,2)
            # chi2 66923.0714571091
            # chi2_H0 66952.78313085782
            # plt.plot(cumchi2-cumchi2_H0,label="cumchi2-cumchi2_H0")
            plt.plot(cumchi2_H1-cumchi2_H0,linestyle="--",label="cumchi2_H1-cumchi2_H0")
            plt.legend()
            plt.figure(4)
            plt.plot(cumchi2_H1,linestyle="--",label="cumchi2_H1")
            plt.plot(cumchi2_H0,linestyle="--",label="cumchi2_H0")
            plt.legend()

            plt.show()


        if 1:
            rvs = np.array([0])
            # rvs = np.linspace(-4000,4000,21,endpoint=True)
            # ra_offset,dec_offset = 0.5,-0.5
            # ras = np.arange(ra_offset-0.0,ra_offset+0.6,0.1)
            # decs = np.arange(dec_offset-0.0,dec_offset+0.6,0.1)
            # ras = np.arange(0,2.5,0.1)
            # decs = np.arange(-2.0,1.0,0.1)
            # ras = np.arange(-2.5,2.5,0.1)
            # decs = np.arange(-3.0,2.0,0.1)
            ras = np.arange(-2.5,2.5,0.05)
            decs = np.arange(-3.0,2.0,0.05)
            # ras = np.arange(-2.0,-1.5,0.1)
            # decs = np.array([0])
            if 0:
                import cProfile
                import pstats
                cProfile.run("grid_search([rvs,ras,decs],dataobj,fm_func,fm_paras,numthreads=None)",
                             '../profiling_results')
                # Load the profiling results into a pstats object
                profiler = pstats.Stats('profiling_results')
                # Sort the results by tottime (total time spent in a function)
                profiler.sort_stats('cumtime')
                # Print the sorted profiling results
                profiler.print_stats()
                exit()
            # log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,ras,decs],dataobj,fm_func,fm_paras,numthreads=None)
            log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,ras,decs],dataobj,fm_func,fm_paras,numthreads=numthreads,computeH0=False)
            N_linpara = linparas.shape[-1]

            import datetime
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

            outoftheoven_filename = os.path.join(out_dir,formatted_datetime+"_"+os.path.basename(filename).replace(".fits","_out.fits"))
            print(outoftheoven_filename)
            with h5py.File(outoftheoven_filename, 'w') as hf:
                hf.create_dataset("rvs", data=rvs)
                hf.create_dataset("ras", data=ras)
                hf.create_dataset("decs", data=decs)
                hf.create_dataset("log_prob", data=log_prob)
                hf.create_dataset("log_prob_H0", data=log_prob_H0)
                hf.create_dataset("rchi2", data=rchi2)
                hf.create_dataset("linparas", data=linparas)
                hf.create_dataset("linparas_err", data=linparas_err)
        else:
            print(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
            outoftheoven_filelist = glob(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
            outoftheoven_filelist.sort()
            outoftheoven_filename = outoftheoven_filelist[-1]
            print(outoftheoven_filename)
            # outoftheoven_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/out/20230417_231940_jw01414004001_02101_00001_nrs2_cal_out.fits"
            with h5py.File(outoftheoven_filename, 'r') as hf:
                rvs = np.array(hf.get("rvs"))
                ras = np.array(hf.get("ras"))
                decs = np.array(hf.get("decs"))
                log_prob = np.array(hf.get("log_prob"))
                log_prob_H0 = np.array(hf.get("log_prob_H0"))
                rchi2 = np.array(hf.get("rchi2"))
                linparas = np.array(hf.get("linparas"))
                linparas_err = np.array(hf.get("linparas_err"))

    mypool.close()
    mypool.join()

    k=0
    # k,l,m = np.unravel_index(np.nanargmax(log_prob-log_prob_H0),log_prob.shape)
    # print("best fit parameters: rv={0},y={1},x={2}".format(rvs[k],ras[l],decs[m]) )
    # print(np.nanmax(log_prob-log_prob_H0))
    # best_log_prob,best_log_prob_H0,_,_,_ = grid_search([[rvs[k]], [ras[l]], [decs[m]]], dataobj, fm_func, fm_paras, numthreads=None)
    # print(best_log_prob-best_log_prob_H0)

    linparas = np.swapaxes(linparas, 1, 2)
    linparas_err = np.swapaxes(linparas_err, 1, 2)
    log_prob = np.swapaxes(log_prob, 1, 2)
    log_prob_H0 = np.swapaxes(log_prob_H0, 1, 2)
    rchi2 = np.swapaxes(rchi2, 1, 2)

    dra=ras[1]-ras[0]
    ddec=decs[1]-decs[0]
    ras_grid,decs_grid = np.meshgrid(ras,decs)
    rs_grid = np.sqrt(ras_grid**2+decs_grid**2)
    print(ra_offset,dec_offset)
    rs_comp_grid = np.sqrt((ras_grid-ra_offset)**2+(decs_grid-dec_offset)**2)
    # plt.imshow(rs_comp_grid,origin="lower")
    # plt.show()

    plt.figure(1)
    plt.subplot(1,3,1)
    plt.title("SNR map")
    snr_map = linparas[k,:,:,0]/linparas_err[k,:,:,0]
    # print("SNR std",np.nanmax(snr_map[np.where(rs_comp_grid>0.4)]))
    print("SNR std",np.nanstd(snr_map[np.where((rs_comp_grid>0.7)*np.isfinite(snr_map))]))
    plt.imshow(snr_map,origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    plt.clim([-2,5])
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    # plt.plot(out[:,0,0,2])
    plt.xlabel("dRA (as)")
    plt.ylabel("ddec (as)")

    contrast_5sig  = 5*linparas_err[k,:,:,0]/HD19467_flux_MJy[photfilter_name]
    print(HD19467_flux_MJy[photfilter_name])
    nan_mask_boxsize=2
    contrast_5sig[np.where(np.isnan(correlate2d(contrast_5sig,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan

    plt.subplot(1,3,2)
    plt.title("Flux map")
    plt.imshow(linparas[k,:,:,0],origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    cbar = plt.colorbar()
    cbar.set_label("planet flux MJy")
    plt.xlabel("dRA (as)")
    plt.ylabel("ddec (as)")
    plt.subplot(1,3,3)
    plt.title("5-$\sigma$ Sensitivity 2D {0}".format(photfilter_name))
    plt.imshow(np.log10(contrast_5sig),origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    # plt.clim([0,100])
    plt.xlabel("dRA (as)")
    plt.ylabel("ddec (as)")
    cbar = plt.colorbar()
    cbar.set_label("5-$\sigma$ Flux ratio log10 ({0})".format(photfilter_name))

    plt.figure(3)
    plt.title("5-$\sigma$ Sensitivity 1D")
    plt.scatter(rs_grid,contrast_5sig)
    plt.yscale("log")
    plt.xlabel("Separation (as)")
    plt.ylabel("5-$\sigma$ Flux ratio ({0})".format(photfilter_name))

    plt.figure(4)
    snr_map_masked = copy(snr_map)
    snr_map_masked[np.where((rs_comp_grid < 0.7))] = np.nan

    # Create a histogram using the hist function from NumPy
    hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20*3, range=(-10, 10))#, bins=256, range=(0, 256)
    bin_centers = (bins[1::]+bins[0:np.size(bins)-1])/2.
    plt.plot(bin_centers,hist/(np.nansum(hist)*(bins[1]-bins[0])),label="snr map")
    plt.plot(bin_centers,1/np.sqrt(2*np.pi)*np.exp(-0.5*(bin_centers-0.0)**2),color="black",linestyle="--",label="Gaussian")
    plt.yscale("log")
    plt.ylim([1e-4,1])
    plt.legend()
    plt.show()




    exit()