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
from breads.fm.hc_splinefm_jwst_nirspec import hc_splinefm_jwst_nirspec

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
    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240225_out_fm/RVs/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
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
    # RA Dec offset of the companion HD19467B
    ra_offset = -1332.871/1000. # ra offset in as
    dec_offset = -875.528/1000. # dec offset in as
        # Offsets for HD 19467 B from https://www.whereistheplanet.com/
        # RA Offset = -1332.871 +/- 10.886 mas
        # Dec Offset = -875.528 +/- 12.360 mas
        # Separation = 1593.703 +/- 9.530 mas
        # PA = 236.712 +/- 0.483 deg
        # Reference: Brandt et al. 2021
    # Flux Calibration parameters
    flux_calib_paras = [-0.03864459,  1.09360589]
    ####################


    mypool = mp.Pool(processes=numthreads)

    for filename in filelist:
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
        from breads.instruments.jwstnirspec_cal import where_point_source
        where2mask = where_point_source(dataobj,(ra_offset,dec_offset),0.4)
        tmp_badpixels = copy(dataobj.bad_pixels)
        tmp_badpixels[where2mask] = np.nan
        subtracted_im,star_model,spline_paras,_ = dataobj.reload_starsubtraction()

        if 1:
            from breads.instruments.jwstnirspec_cal import PCA_wvs_axis
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

        if 1: # read and normalize BTSettl model grid used in FM
            minwv,maxwv= np.min(dataobj.wavelengths),np.max(dataobj.wavelengths)
            with h5py.File(os.path.join(external_dir,"BT-Settlchris_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5"), 'r') as hf:
                grid_specs = np.array(hf.get("spec"))
                grid_temps = np.array(hf.get("temps"))
                grid_loggs = np.array(hf.get("loggs"))
                grid_wvs = np.array(hf.get("wvs"))

            # add flux calibration correction because we are fitting a pure WebbPSF model later, which comes with systematics
            grid_specs = grid_specs / np.polyval(flux_calib_paras, grid_wvs)[None,None,:]

            grid_dwvs = grid_wvs[1::]-grid_wvs[0:np.size(grid_wvs)-1]
            grid_dwvs = np.insert(grid_dwvs,0,grid_dwvs[0])
            filter_norm = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs))
            Flambda = np.nansum((grid_dwvs*u.um)[None,None,:]*photfilter_f(grid_wvs)[None,None,:]*(grid_specs*u.W*u.m**-2/u.um),axis=2)/filter_norm
            Fnu = Flambda*(photfilter_wv0*u.um)**2/const.c # from Flambda back to Fnu
            grid_specs = grid_specs/Fnu[:,:,None].to(u.MJy).value

            myinterpgrid = RegularGridInterpolator((grid_temps,grid_loggs),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
            # teff,logg,vsini,rv,dra_comp,ddec_comp = None,None,0.0,6.953,ra_offset,dec_offset
            teff,logg,vsini,rv,dra_comp,ddec_comp = 940,5.0,0.0,None,ra_offset,dec_offset
            fix_parameters = [teff,logg,vsini,rv,dra_comp,ddec_comp]
        # Definition of the priors
        if 1:
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

        from breads.fm.hc_atmgrid_splinefm_jwst_nirspec_cal import hc_atmgrid_splinefm_jwst_nirspec_cal
        fm_paras = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"star_func":dataobj.star_func,
                    "radius_as":0.1,"badpixfraction":0.75,"nodes":x_nodes,"fix_parameters":fix_parameters,
                    "wvs_KLs_f":wvs_KLs_f_list,"regularization":"user","reg_mean_map":reg_mean_map,"reg_std_map":reg_std_map}
        #"wvs_KLs_f":wvs_KLs_f_list
        #"regularization":"user"
        fm_func = hc_atmgrid_splinefm_jwst_nirspec_cal

        # /!\ Optional but recommended
        # Test the forward model for a fixed value of the non linear parameter.
        # Make sure it does not crash and look the way you want
        if 0:
            nonlin_paras = [6]
            # d is the data vector a the specified location
            # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
            # s is the vector of uncertainties corresponding to d
            d, M, s,extra_outputs = fm_func(nonlin_paras,dataobj,return_extra_outputs=True,**fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            w = extra_outputs["wvs"]
            x = extra_outputs["ras"]
            y = extra_outputs["decs"]
            rows = extra_outputs["rows"]
            unique_rows = np.unique(rows)
            shifted_w = w+(wv_sampling[-1]-wv_sampling[0])*(rows-np.nanmin(unique_rows))
            if fm_paras["regularization"] is not None:
                d_reg,s_reg = extra_outputs["regularization"]
                reg_wvs = extra_outputs["regularization_wvs"]
                reg_rows = extra_outputs["regularization_rows"]
                shifted_reg_wvs = reg_wvs+(wv_sampling[-1]-wv_sampling[0])*(reg_rows-np.nanmin(unique_rows))

            # print(np.unique(where_finite[0]))
            # mask_comp = np.zeros(dataobj.data.shape)
            # mask_comp[where_finite] = 1


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
                                                                         scale_noise=False,marginalize_noise_scaling=False,debug=False)
            paras = linparas[validpara]
            print("best fit", linparas[0:5])
            print("best fit err",linparas_err[0:5])
            print("best fit snr",linparas[0:5]/linparas_err[0:5])
            print("rchi2",rchi2)
            print("log_prob",log_prob)
            # exit()
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
            if fm_paras["regularization"] is not None:
                where_even_rows = np.where((reg_rows % 2)==0)
                plt.errorbar(shifted_reg_wvs[where_even_rows],d_reg[where_even_rows],yerr=s_reg[where_even_rows],label="even rows prior")
                where_odd_rows = np.where((reg_rows % 2)==1)
                plt.errorbar(shifted_reg_wvs[where_odd_rows],d_reg[where_odd_rows],yerr=s_reg[where_odd_rows],label="odd rows prior")
            plt.ylabel("Flux (MJy)")
            plt.xlabel("Column pixels")
            plt.legend()
            # plt.show()

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
            rvs = np.arange(-20,20.0001,0.1)
            log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs],dataobj,fm_func,fm_paras,numthreads=numthreads,computeH0=False,scale_noise=True)
            N_linpara = linparas.shape[-1]

            import datetime
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y%m%d_%H%M%S")


            outoftheoven_filename = os.path.join(out_dir,formatted_datetime+"_"+os.path.basename(filename).replace(".fits","_out.h5"))
            print(outoftheoven_filename)
            with h5py.File(outoftheoven_filename, 'w') as hf:
                hf.create_dataset("rvs", data=rvs)
                hf.create_dataset("log_prob", data=log_prob)
                hf.create_dataset("log_prob_H0", data=log_prob_H0)
                hf.create_dataset("rchi2", data=rchi2)
                hf.create_dataset("linparas", data=linparas)
                hf.create_dataset("linparas_err", data=linparas_err)

            plt.plot(rvs,np.exp(log_prob- np.nanmax(log_prob)),label=detector)
            # plt.show()
        else:
            print(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.h5")))
            outoftheoven_filelist = glob(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.h5")))
            outoftheoven_filelist.sort()
            outoftheoven_filename = outoftheoven_filelist[-1]
            print(outoftheoven_filename)
            # outoftheoven_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/out/20230417_231940_jw01414004001_02101_00001_nrs2_cal_out.fits"
            with h5py.File(outoftheoven_filename, 'r') as hf:
                rvs = np.array(hf.get("rvs"))
                log_prob = np.array(hf.get("log_prob"))
                log_prob_H0 = np.array(hf.get("log_prob_H0"))
                rchi2 = np.array(hf.get("rchi2"))
                linparas = np.array(hf.get("linparas"))
                linparas_err = np.array(hf.get("linparas_err"))
            plt.plot(rvs,np.exp(log_prob- np.nanmax(log_prob)),label=detector)

    mypool.close()
    mypool.join()

    plt.xlabel("RV (km/s)")
    plt.show()
    # exit()

    print(linparas[:,:,0])
    print(linparas_err[:,:,0])
    print(log_prob)
    k,l = np.unravel_index(np.nanargmax(log_prob),log_prob.shape)
    print("best fit parameters: teff={0},logg={1}".format(teffs[k],loggs[l]) )
    print(np.nanmax(log_prob))
    best_log_prob,best_log_prob_H0,_,_,_ = grid_search([[teffs[k]], [loggs[l]]], dataobj, fm_func, fm_paras, numthreads=None)
    print(best_log_prob)

    print(linparas.shape)
    linparas = np.swapaxes(linparas, 0, 1)
    linparas_err = np.swapaxes(linparas_err, 0, 1)
    log_prob = np.swapaxes(log_prob, 0, 1)
    log_prob_H0 = np.swapaxes(log_prob_H0, 0, 1)
    rchi2 = np.swapaxes(rchi2, 0, 1)

    dTeff=teffs[1]-teffs[0]
    dlogg=loggs[1]-loggs[0]
    Teff_grid,logg_grid = np.meshgrid(dTeff,dlogg)

    plt.figure(1)
    plt.subplot(1,3,1)
    plt.title("SNR map")
    snr_map = linparas[:,:,0]/linparas_err[:,:,0]
    aspect = (teffs[-1]+dTeff/2.-(teffs[0]-dTeff/2.))/(loggs[-1]+dlogg/2.-(loggs[0]-dlogg/2.))
    plt.imshow(snr_map,origin="lower",extent=[teffs[0]-dTeff/2.,teffs[-1]+dTeff/2.,loggs[0]-dlogg/2.,loggs[-1]+dlogg/2.],aspect=aspect)
    plt.clim([-2,5])
    cbar = plt.colorbar()
    cbar.set_label("SNR")
    # plt.plot(out[:,0,0,2])
    plt.xlabel("Teff (K)")
    plt.ylabel("log(g) ")

    plt.subplot(1,3,2)
    plt.title("Flux map")
    plt.imshow(linparas[:,:,0],origin="lower",extent=[teffs[0]-dTeff/2.,teffs[-1]+dTeff/2.,loggs[0]-dlogg/2.,loggs[-1]+dlogg/2.],aspect=aspect)
    cbar = plt.colorbar()
    cbar.set_label("planet flux MJy")
    plt.xlabel("Teff (K)")
    plt.ylabel("log(g) ")

    plt.subplot(1,3,3)
    plt.title("5-$\sigma$ Sensitivity 2D {0}".format(photfilter_name))
    plt.imshow(log_prob- np.nanmax(log_prob),origin="lower",extent=[teffs[0]-dTeff/2.,teffs[-1]+dTeff/2.,loggs[0]-dlogg/2.,loggs[-1]+dlogg/2.],aspect=aspect)
    # plt.clim([0,100])
    plt.xlabel("Teff (K)")
    plt.ylabel("log(g) ")
    cbar = plt.colorbar()
    cbar.set_label("log(prob) ({0})".format(photfilter_name))

    plt.show()




    exit()