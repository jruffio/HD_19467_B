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
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"

    numthreads = 20
    N_KL = 3#0#3

    # RA Dec offset of the companion
    ra_offset = -1332.871/1000. # ra offset in as
    dec_offset = -875.528/1000. # dec offset in as
    # ra_offset = -1.38 # ra offset in as
    # dec_offset = -0.92 # dec offset in as
    # Offsets for HD 19467 B
    # RA Offset = -1332.871 +/- 10.886 mas
    # Dec Offset = -875.528 +/- 12.360 mas
    # Separation = 1593.703 +/- 9.530 mas
    # PA = 236.712 +/- 0.483 deg
    # Reference: Brandt et al. 2021

    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240127_out/xy/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    utils_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240127_utils/"
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
    external_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/"

    # Choose to reduce either detector (NRS1 or NRS2)
    detector,photfilter_name,wv_sampling = "nrs1", "F360M", np.arange(2.859509, 4.1012874, 0.0006763935)
    # detector,photfilter_name,wv_sampling= "nrs2", "F460M",np.arange(4.081285,5.278689,0.0006656647)

    # Definition of the nodes for the spline
    if detector == "nrs1":
        dw1,dw2 = 0.02,0.04
        l0,l2 = 2.859509-dw1, 4.1012874
        l1 = (l0+l2)/2
        x1_nodes = np.arange(l0,l1, dw1)
        l1 = x1_nodes[-1]+dw2
        x2_nodes = np.arange(l1,l2, dw2)
        x_nodes = np.concatenate([x1_nodes,x2_nodes])
    elif detector == "nrs2":
        dw1,dw2 = 0.04,0.02
        l0,l2 = 4.081285,5.278689+dw2
        l1 = (l0+l2)/2
        x1_nodes = np.arange(l0,l1, dw1)
        l1 = x1_nodes[-1]+dw2
        x2_nodes = np.arange(l1,l2, dw2)
        x_nodes = np.concatenate([x1_nodes,x2_nodes])
    print(x_nodes)
    print(np.size(x_nodes))
    # exit()

    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_"+detector+"_cal.fits")
    #"/stow/jruffio/data/JWST/nirspec/HD_19467/20240125_MAST_1414/MAST_2024-01-25T1449/JWST/jw01414001001_02101_00001_nrs2/jw01414001001_02101_00001_nrs2_cal.fits"
    # filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/20240125_MAST_1414/MAST_2024-01-25T1449/JWST/jw01414004001_02101_000*_"+detector+"/jw01414004001_02101_*_"+detector+"_cal.fits")
    # filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/stage2/jw01414004001_02101_*_"+detector+"_cal.fits")
    filelist.sort()
    print("I found the following files:")
    for fid,filename in enumerate(filelist):
        print(filename)
    print("N files: {0}".format(len(filelist)))

    # Define
    HD19467_flux_MJy = {"F250M":3.51e-6, # in MJy, Ref Greenbaum+2023
                         "F300M":2.63e-6,
                         "F335M":2.10e-6,
                         "F360M":1.82e-6,
                         "F410M":1.49e-6,
                         "F430M":1.36e-6,
                         "F460M":1.12e-6}

    try:
        photfilter = os.path.join(external_dir,"JWST_NIRCam."+photfilter_name+".dat")
        filter_arr = np.loadtxt(photfilter)
    except:
        photfilter = os.path.join(external_dir,"Keck_NIRC2."+photfilter_name+".dat")
        filter_arr = np.loadtxt(photfilter)
    trans_wvs = filter_arr[:,0]/1e4
    trans = filter_arr[:,1]
    photfilter_f = interp1d(trans_wvs,trans,bounds_error=False,fill_value=0)
    photfilter_wv0 = np.nansum(trans_wvs*photfilter_f(trans_wvs))/np.nansum(photfilter_f(trans_wvs))
    bandpass = np.where(photfilter_f(trans_wvs)/np.nanmax(photfilter_f(trans_wvs))>0.01)
    photfilter_wvmin,photfilter_wvmax = trans_wvs[bandpass[0][0]],trans_wvs[bandpass[0][-1]]
    print(photfilter_wvmin,photfilter_wvmax)
    # exit()
    # plt.plot(trans_wvs,trans)
    # plt.show()

    mypool = mp.Pool(processes=numthreads)

    preproc_task_list = []
    preproc_task_list.append(["compute_med_filt_badpix", {"window_size": 50, "mad_threshold": 50}, True, True])
    preproc_task_list.append(["compute_coordinates_arrays"])
    preproc_task_list.append(["convert_MJy_per_sr_to_MJy"]) # old reduction, already in MJy
    ra_corr,dec_corr = -0.1285876002234175, - 0.06997868615326872
    preproc_task_list.append(["apply_coords_offset",{"coords_offset":(ra_corr,dec_corr)}]) #-0.11584366936455087, 0.07189009712128012
    preproc_task_list.append(["compute_webbpsf_model",
                              {"wv_sampling": wv_sampling, "image_mask": None, "pixelscale": 0.1, "oversample": 10,
                               "parallelize": False, "mppool": mypool}, True, True])
    # preproc_task_list.append(["compute_new_coords_from_webbPSFfit", {"IWA": 0.2, "OWA": 1.0}, True, True])
    preproc_task_list.append(["compute_charge_bleeding_mask", {"threshold2mask": 0.15}])
    preproc_task_list.append(["compute_starspectrum_contnorm", {"x_nodes": x_nodes, "mppool": mypool}, True, True])
    preproc_task_list.append(["compute_starsubtraction",{"mppool":mypool},True,True])
    preproc_task_list.append(["compute_interpdata_regwvs",{"wv_sampling":wv_sampling},True,True])

    for filename in filelist[0::]:
        print(filename)

        dataobj = JWSTNirspec_cal(filename, crds_dir=crds_dir, utils_dir=utils_dir,
                                  save_utils=True,load_utils=True,preproc_task_list=preproc_task_list)

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
            # for k in range(KLs_left.shape[1]):
            #     KL_f = interp1d(KLs_wvs_left,KLs_left[:,k], bounds_error=False, fill_value=0.0,kind="cubic")
            #     wvs_KLs_f_list.append(KL_f)
            if detector == "nrs1":
                for k in range(KLs_right.shape[1]):
                    KL_f = interp1d(KLs_wvs_right,KLs_right[:,k], bounds_error=False, fill_value=0.0,kind="cubic")
                    wvs_KLs_f_list.append(KL_f)
            else:
                wvs_KLs_f_list = None
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

        if 1: # read and normalize model grid
            RDI_spec_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/20230705_HD19467b_RDI_1dspectrum.fits"
            hdulist_sc = fits.open(RDI_spec_filename)
            grid_wvs = hdulist_sc[0].data
            grid_specs = ((hdulist_sc[1].data[None,None,:]*u.MJy)*(const.c/(grid_wvs*u.um)**2)).to(u.W*u.m**-2/u.um).value
            err = hdulist_sc[2].data

            grid_dwvs = grid_wvs[1::]-grid_wvs[0:np.size(grid_wvs)-1]
            grid_dwvs = np.insert(grid_dwvs,0,grid_dwvs[0])
            filter_norm = np.nansum((grid_dwvs*u.um)*photfilter_f(grid_wvs))
            Flambda = np.nansum((grid_dwvs*u.um)[None,None,:]*photfilter_f(grid_wvs)[None,None,:]*(grid_specs*u.W*u.m**-2/u.um),axis=2)/filter_norm
            Fnu = Flambda*(photfilter_wv0*u.um)**2/const.c # from Flambda back to Fnu
            grid_specs = grid_specs/Fnu[:,:,None].to(u.MJy).value

            # The code expects a model grid but we have a single spectrum, so we are creating a dumb grid but fixing the parameters anyway...
            grid_specs = np.tile(grid_specs,(2,2,1))
            # grid_specs[np.where(np.isnan(grid_specs))] = 0
            myinterpgrid = RegularGridInterpolator(([-1,1],[-1,1]),grid_specs,method="linear",bounds_error=False,fill_value=np.nan)
            teff,logg,vsini,rv,dra_comp,ddec_comp = 0.0,0.0,0.0,None,None,None
            fix_parameters = [teff,logg,vsini,rv,dra_comp,ddec_comp]

        if 1: # Definition of the prior
            subtracted_im,star_model,spline_paras,x_nodes = dataobj.reload_starsubtraction()

            wherenan = np.where(np.isnan(spline_paras))
            reg_mean_map = copy(spline_paras)
            reg_mean_map[wherenan] = np.tile(np.nanmedian(spline_paras, axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
            reg_std_map = np.abs(spline_paras)
            reg_std_map[wherenan] = np.tile(np.nanmax(np.abs(spline_paras), axis=1)[:, None], (1, spline_paras.shape[1]))[wherenan]
            reg_std_map = reg_std_map
            reg_std_map = np.clip(reg_std_map, 1e-11, np.inf)

        from breads.fm.hc_atmgrid_splinefm_jwst_nirspec_cal import hc_atmgrid_splinefm_jwst_nirspec_cal
        fm_paras = {"atm_grid":myinterpgrid,"atm_grid_wvs":grid_wvs,"star_func":dataobj.star_func,
                    "radius_as":0.1,"badpixfraction":0.75,"nodes":x_nodes,"fix_parameters":fix_parameters,
                    "wvs_KLs_f":wvs_KLs_f_list,"regularization":"user","reg_mean_map":reg_mean_map,"reg_std_map":reg_std_map}
        fm_func = hc_atmgrid_splinefm_jwst_nirspec_cal

        # /!\ Optional but recommended
        # Test the forward model for a fixed value of the non linear parameter.
        # Make sure it does not crash and look the way you want
        if 1:
            # fm_paras["fix_parameters"]= [None,None,None,sc_fib]
            print(ra_offset,dec_offset)
            # nonlin_paras = [0.0,ra_offset,dec_offset] # x (pix),y (pix), rv (km/s)
            nonlin_paras = [0, 0.4,-1.95] # x (pix),y (pix), rv (km/s)
            # nonlin_paras = [0, -1.4, -1.4] # x (pix),y (pix), rv (km/s)
            # d is the data vector a the specified location
            # M is the linear component of the model. M is a function of the non linear parameters x,y,rv
            # s is the vector of uncertainties corresponding to d
            d, M, s,extra_outputs = fm_func(nonlin_paras,dataobj,return_where_finite=True,**fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            print(np.unique(where_finite[0]))
            mask_comp = np.zeros(dataobj.data.shape)
            mask_comp[where_finite] = 1

            wvs_im = dataobj.wavelengths
            cols_im = np.tile(np.arange(wvs_im.shape[1])[None,:],(wvs_im.shape[0],1))
            d_wvs = wvs_im[where_finite]
            d_cols = cols_im[where_finite]

            validpara = np.where(np.max(np.abs(M),axis=0)!=0)
            M = M[:,validpara[0]]

            d = d / s
            M = M / s[:, None]

            from breads.fit import fitfm
            log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm(nonlin_paras, dataobj, fm_func, fm_paras,computeH0 = True,bounds = None,residuals=None,residuals_H0=None)
            paras = linparas[validpara]
            print("best fit", linparas[0:5])
            print("best fit err",linparas_err[0:5])
            print("best fit snr",linparas[0:5]/linparas_err[0:5])
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
            plt.subplot(4,1,1)
            plt.plot(d*s,label="data")
            plt.plot(m*s,label="Combined model")
            plt.plot(paras[0]*M[:,0]*s,label="planet model")
            plt.plot((m-paras[0]*M[:,0])*s,label="starlight model")
            plt.ylabel("Flux (MJy)")
            plt.xlabel("Column pixels")
            plt.legend()

            plt.subplot(4,1,2)
            plt.plot(r,label="Residuals")
            plt.fill_between(np.arange(np.size(s)),-1,1,label="Error",alpha=0.5)
            r_std = np.nanstd(r)
            plt.ylim([-10*r_std,10*r_std])
            plt.ylabel("Flux (MJy)")
            plt.xlabel("Column pixels")
            plt.legend()

            plt.subplot(4,1,3)
            plt.plot(M[:,0],label="planet model")

            plt.subplot(4,1,4)
            for k in range(np.min([50,M.shape[-1]-1])):
                plt.plot(M[:,k+1],label="starlight model {0}".format(k+1))
            plt.legend()

            plt.figure(2)
            plt.subplot(3,1,1)
            plt.scatter(d_wvs,res,alpha=0.5,s=0.1)
            plt.xlabel("Wavelength (um)")
            plt.subplot(3,1,2)
            plt.scatter(d_cols,res,alpha=0.5,s=0.1)
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
            log_prob,log_prob_H0,rchi2,linparas,linparas_err = grid_search([rvs,ras,decs],dataobj,fm_func,fm_paras,numthreads=numthreads)
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

    k,l,m = np.unravel_index(np.nanargmax(log_prob-log_prob_H0),log_prob.shape)
    print("best fit parameters: rv={0},y={1},x={2}".format(rvs[k],ras[l],decs[m]) )
    print(np.nanmax(log_prob-log_prob_H0))
    best_log_prob,best_log_prob_H0,_,_,_ = grid_search([[rvs[k]], [ras[l]], [decs[m]]], dataobj, fm_func, fm_paras, numthreads=None)
    print(best_log_prob-best_log_prob_H0)

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