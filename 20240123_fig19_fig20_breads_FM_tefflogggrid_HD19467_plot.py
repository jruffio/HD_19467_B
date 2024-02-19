import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import h5py
from  scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const
import astropy.io.fits as fits

from breads.utils import get_err_from_posterior

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar
import matplotlib.patheffects as PathEffects
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})


if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    ####################
    ## To be modified
    ####################
    # directory where the teff/logg grid fits are saved: the outputs from breads.grid_search
    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240216_out_fm/tefflogg_FMRDI/"
    # directory where the fine sampling grid fits are saved (for error bar calculation)
    out_dir_local = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240216_out_fm/tefflogg_FMRDI_local/"
    # out_dir_local = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240216_out_fm/tefflogg_looseFMRDI_local/"
    ####################
    # for plotting
    fontsize = 12
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # External_dir should external files like the NIRCam filters
    external_dir = "/stow/jruffio/data/JWST/external/"
    # output dir for images
    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
    if not os.path.exists(out_png):
        os.makedirs(out_png)
    # Science data: List of stage 2 cal.fits files
    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
    color_list = ["#ff9900","#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
    ####################


    if 1:
        fluxMJy_bestfit = np.zeros((2,9))+np.nan
        fluxMJy_err = np.zeros((2,9))+np.nan
        teff_bestfit = np.zeros((2,9))+np.nan
        teff_errm = np.zeros((2,9))+np.nan
        teff_errp = np.zeros((2,9))+np.nan
        for detid, detector in enumerate(["nrs1","nrs2"]):
            filelist_nrs = [filename for filename in filelist if detector in filename]
            for fid,filename in enumerate(filelist_nrs):

                print(os.path.join(out_dir_local,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
                outoftheoven_filelist = glob(os.path.join(out_dir_local,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
                outoftheoven_filelist.sort()
                if len(outoftheoven_filelist)==0:
                    continue
                print(outoftheoven_filelist)
                outoftheoven_filename = outoftheoven_filelist[-1]
                with h5py.File(outoftheoven_filename, 'r') as hf:
                    teffs = np.array(hf.get("teffs"))
                    loggs = np.array(hf.get("loggs"))
                    log_prob = np.array(hf.get("log_prob"))
                    log_prob_H0 = np.array(hf.get("log_prob_H0"))
                    rchi2 = np.array(hf.get("rchi2"))
                    linparas = np.array(hf.get("linparas"))
                    linparas_err = np.array(hf.get("linparas_err"))

                k,l = np.unravel_index(np.nanargmax(log_prob),log_prob.shape)
                print("best fit parameters: teff={0},logg={1}".format(teffs[k],loggs[l]) )
                print(np.nanmax(log_prob))
                # best_log_prob,best_log_prob_H0,_,_,_ = grid_search([[teffs[k]], [loggs[l]]], dataobj, fm_func, fm_paras, numthreads=None)
                # print(best_log_prob)

                print(linparas.shape)
                linparas = np.swapaxes(linparas, 0, 1)
                linparas_err = np.swapaxes(linparas_err, 0, 1)
                log_prob = np.swapaxes(log_prob, 0, 1)
                log_prob_H0 = np.swapaxes(log_prob_H0, 0, 1)
                rchi2 = np.swapaxes(rchi2, 0, 1)

                dTeff=teffs[1]-teffs[0]
                dlogg=loggs[1]-loggs[0]
                Teff_grid,logg_grid = np.meshgrid(dTeff,dlogg)

                print(log_prob.shape)
                teff, err_m,err_p = get_err_from_posterior(teffs,np.exp(log_prob[l,:]- np.nanmax(log_prob[l,:])))
                print(teff, err_m,err_p)
                teff_bestfit[detid,fid] = teff
                teff_errm[detid,fid] = err_m
                teff_errp[detid,fid] = err_p
                fluxMJy_bestfit[detid,fid] = linparas[l,k,0]
                fluxMJy_err[detid,fid] = linparas_err[l,k,0]
    # exit()
    # teff_bestfit[:,6] = np.nan
    # teff_errm[:,6] = np.nan
    # teff_errm[:,6] = np.nan
    print(fluxMJy_bestfit)
    teff_err = np.nanmax(np.concatenate((teff_errm[:,:,None],teff_errm[:,:,None]),axis=2),axis=2)
    fluxMJy_bestfit_combined = np.nansum(fluxMJy_bestfit / fluxMJy_err ** 2, axis=1) / np.nansum(1 / fluxMJy_err ** 2, axis=1)
    print("fluxMJy_bestfit_combined",fluxMJy_bestfit_combined)
    fluxMJy_err_combined = np.sqrt(1/np.nansum(1/fluxMJy_err**2,axis=1))
    print("fluxMJy_err_combined",fluxMJy_err_combined)
    teff_bestfit_combined = np.nansum(teff_bestfit / teff_err ** 2, axis=1) / np.nansum(1 / teff_err ** 2, axis=1)
    print("teff_bestfit_combined",teff_bestfit_combined)
    teff_err_combined = np.sqrt(1/np.nansum(1/teff_err**2,axis=1))
    print("teff_err_combined",teff_err_combined)
    print(np.nanmean(teff_err,axis=1))
    print(np.nanstd(teff_bestfit,axis=1))
# [949.50755561 972.12973409]
# [4.10243823 2.75794756]
# [48.29021065 16.47575847]
#     exit()




    plt.figure(1,figsize=(12,5.5))
    gs = gridspec.GridSpec(2,4, height_ratios=[0.05,1], width_ratios=[1,1,0.3,1.0])
    gs.update(left=0.05, right=0.95, bottom=0.19, top=0.85, wspace=0.0, hspace=0.0)
    for detid, detector in enumerate(["nrs1","nrs2"]):
    # for detector in ["nrs2"]:
        filelist_nrs = [filename for filename in filelist if detector in filename]
        filename = filelist_nrs[0]

        print(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
        outoftheoven_filelist = glob(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
        outoftheoven_filelist.sort()
        print(outoftheoven_filelist)
        outoftheoven_filename = outoftheoven_filelist[-1]
        # print(outoftheoven_filename)
        # exit()
        # outoftheoven_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/out/20230417_231940_jw01414004001_02101_00001_nrs2_cal_out.fits"
        with h5py.File(outoftheoven_filename, 'r') as hf:
            teffs = np.array(hf.get("teffs"))
            loggs = np.array(hf.get("loggs"))
            log_prob = np.array(hf.get("log_prob"))
            log_prob_H0 = np.array(hf.get("log_prob_H0"))
            rchi2 = np.array(hf.get("rchi2"))
            linparas = np.array(hf.get("linparas"))
            linparas_err = np.array(hf.get("linparas_err"))
        # continue

        # print(linparas[:,:,0])
        # print(linparas_err[:,:,0])
        # print(log_prob)
        k,l = np.unravel_index(np.nanargmax(log_prob),log_prob.shape)
        print("best fit parameters: teff={0},logg={1}".format(teffs[k],loggs[l]) )
        print(np.nanmax(log_prob))
        # best_log_prob,best_log_prob_H0,_,_,_ = grid_search([[teffs[k]], [loggs[l]]], dataobj, fm_func, fm_paras, numthreads=None)
        # print(best_log_prob)

        print(linparas.shape)
        linparas = np.swapaxes(linparas, 0, 1)
        linparas_err = np.swapaxes(linparas_err, 0, 1)
        log_prob = np.swapaxes(log_prob, 0, 1)
        log_prob_H0 = np.swapaxes(log_prob_H0, 0, 1)
        rchi2 = np.swapaxes(rchi2, 0, 1)
        print(np.nanmedian(rchi2))

        dTeff=teffs[1]-teffs[0]
        dlogg=loggs[1]-loggs[0]
        Teff_grid,logg_grid = np.meshgrid(dTeff,dlogg)

        fontsize=12
        # gs.update(left=0.00, right=1, bottom=0.0, top=1, wspace=0.0, hspace=0.0)
        ax1 = plt.subplot(gs[1, detid])
        aspect = (teffs[-1]+dTeff/2.-(teffs[0]-dTeff/2.))/(loggs[-1]+dlogg/2.-(loggs[0]-dlogg/2.))
        plt1 = plt.imshow(log_prob- np.nanmax(log_prob),origin="lower",extent=[teffs[0]-dTeff/2.,teffs[-1]+dTeff/2.,loggs[0]-dlogg/2.,loggs[-1]+dlogg/2.],aspect=aspect)
        txt = plt.text(0.03, 0.99, detector, fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
        # plt.clim([0,100])
        if detid == 0:
            plt.xlabel(r"T$_{\mathrm{eff}}$ (K)",fontsize=fontsize)
            plt.ylabel(r"ln(g/(1 cm.s$^{-2}$))",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
        else:
            plt.gca().set_yticklabels([])
            plt.xlabel(r"T$_{\mathrm{eff}}$ (K)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.clim([-1000,0])

        cbax = plt.subplot(gs[0, detid])
        cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')
        # cb.set_label(r'test', labelpad=5, fontsize=fontsize)
        cb.set_label(r'ln[$\mathcal{P}$(T$_{\mathrm{eff}}$,log(g)|d))]+cst', labelpad=5, fontsize=fontsize)#
        if detid == 0:
            cb.set_ticks([-1000,-750,-500,-250,0])  # xticks[1::]
        else:
            cb.set_ticks([-750,-500,-250,0])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax1 = plt.subplot(gs[1, 3])
        if detid ==0:
            plt.scatter(np.arange(teff_bestfit.shape[1])+1,teff_bestfit[detid,:],label=detector,marker="x")
        else:
            plt.scatter(np.arange(teff_bestfit.shape[1])+1,teff_bestfit[detid,:],label=detector,marker="o")
        plt.errorbar(np.arange(teff_bestfit.shape[1])+1,teff_bestfit[detid,:],yerr=[teff_errm[detid,:],teff_errp[detid,:]], ls='none')
        plt.ylim([900,1000])
        plt.xlabel("Dither #",fontsize=fontsize)
        plt.ylabel(r"T$_{\mathrm{eff}}$ (K)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.xticks([1,2,3,4,5,6,7,8,9])
        plt.legend(loc="upper left", handlelength=3)

    # plt.tight_layout()
    out_filename = os.path.join(out_png, "logprob_FMRDI.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))


    if 1: # plot figure 20

        RDI_1dspectrum_filename = os.path.join(out_png, "HD19467b_RDI_1dspectrum_Flambda.fits")
        with fits.open(RDI_1dspectrum_filename) as hdulist:
            HD19467B_wvs = hdulist[0].data
            HD19467B_spec_Flambda = hdulist[1].data
            HD19467B_spec_err_Flambda = hdulist[2].data
            HD19467B_spec_errhpf_Flambda = hdulist[3].data

        photfilter = os.path.join(external_dir,"JWST_NIRCam." + "F360M" + ".dat")
        filter_arr = np.loadtxt(photfilter)
        trans_wvs = filter_arr[:, 0] / 1e4
        trans = filter_arr[:, 1]
        photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
        photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
        with h5py.File(os.path.join(external_dir, "BT-Settlchris_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5"), 'r') as hf:
            grid_specs = np.array(hf.get("spec"))
            grid_temps = np.array(hf.get("temps"))
            grid_loggs = np.array(hf.get("loggs"))
            grid_wvs = np.array(hf.get("wvs"))
        grid_dwvs = grid_wvs[1::] - grid_wvs[0:np.size(grid_wvs) - 1]
        grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
        filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs))
        Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs)[None, None, :] * (
                    grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
        Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
        grid_specs = grid_specs / Fnu[:, :, None].to(u.MJy).value
        myinterpgrid_F360M = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear", bounds_error=False,
                                               fill_value=np.nan)

        photfilter = os.path.join(external_dir,"JWST_NIRCam." + "F460M" + ".dat")
        filter_arr = np.loadtxt(photfilter)
        trans_wvs = filter_arr[:, 0] / 1e4
        trans = filter_arr[:, 1]
        photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
        photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
        with h5py.File(os.path.join(external_dir, "BT-Settlchris_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5"), 'r') as hf:
            grid_specs = np.array(hf.get("spec"))
            grid_temps = np.array(hf.get("temps"))
            grid_loggs = np.array(hf.get("loggs"))
            grid_wvs = np.array(hf.get("wvs"))
        grid_dwvs = grid_wvs[1::] - grid_wvs[0:np.size(grid_wvs) - 1]
        grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
        filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs))
        Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs)[None, None, :] * (
                    grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
        Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
        grid_specs = grid_specs / Fnu[:, :, None].to(u.MJy).value
        myinterpgrid_F460M = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear", bounds_error=False,
                                               fill_value=np.nan)

    fontsize=12
    plt.figure(3,figsize=(12,8))
    for nrs_id,(lmin,lmax) in enumerate([(2.85,4.25),(4.05,5.3)]):
        plt.subplot(2,1,nrs_id+1)

        if nrs_id == 0:
            best_fit_planet_model = myinterpgrid_F360M((teff_bestfit_combined[nrs_id], 5.0))
            planet_f = interp1d(grid_wvs, best_fit_planet_model, bounds_error=False, fill_value=0)
            mytext = r"NRS1 FM+RDI Best fit: T$_{\mathrm{eff}}$ = "+"{0:.0f}".format(teff_bestfit_combined[nrs_id])+r"$\pm$"+"{0:.0f}".format(teff_err_combined[nrs_id])+" K ; log(g/(1 cm.s$^{-2}$)) = 5.0"+" ; Flux in F360M = "+"{0:.1f}".format(fluxMJy_bestfit_combined[nrs_id]*1e12)+r"$\pm$"+"{0:.1f}".format(fluxMJy_err_combined[nrs_id]*1e12)+r" $\mu$Jy"
            plt.text(0.01, 0.99,mytext , fontsize=fontsize, ha='left', va='top', color="black",transform=plt.gca().transAxes)  # \n 36-166 $\mu$Jy
        if nrs_id == 1:
            best_fit_planet_model = myinterpgrid_F460M((teff_bestfit_combined[nrs_id], 5.0))
            planet_f = interp1d(grid_wvs, best_fit_planet_model, bounds_error=False, fill_value=0)
            mytext = r"NRS2 FM+RDI Best fit: T$_{\mathrm{eff}}$ = "+"{0:.0f}".format(teff_bestfit_combined[nrs_id])+r"$\pm$"+"{0:.0f}".format(teff_err_combined[nrs_id])+" K ; log(g/(1 cm.s$^{-2}$)) = 5.0"+" ; Flux in F460M = "+"{0:.1f}".format(fluxMJy_bestfit_combined[nrs_id]*1e12)+r"$\pm$"+"{0:.1f}".format(fluxMJy_err_combined[nrs_id]*1e12)+r" $\mu$Jy"
            plt.text(0.01, 0.99, mytext, fontsize=fontsize, ha='left', va='top', color="black",transform=plt.gca().transAxes)  # \n 36-166 $\mu$Jy

        plt.fill_between(HD19467B_wvs,-HD19467B_spec_err_Flambda,+HD19467B_spec_err_Flambda,color=color_list[1],alpha=0.5, lw=0,label="RDI error")
        plt.plot(HD19467B_wvs,HD19467B_spec_Flambda,color=color_list[1],label="RDI spectrum",linewidth=1,linestyle="--")
        plt.plot(HD19467B_wvs, planet_f(HD19467B_wvs) * fluxMJy_bestfit_combined[nrs_id],color=color_list[0],label="FM+RDI best-fit model",linewidth=1,linestyle="-")
        plt.plot(HD19467B_wvs, HD19467B_spec_Flambda-planet_f(HD19467B_wvs) * fluxMJy_bestfit_combined[nrs_id],color="black",label="Difference",linewidth=0.5,linestyle=":")

        plt.xlim([lmin,lmax])
        plt.ylim([-0.7e-17,5.5e-17])
        if nrs_id == 1:
            plt.legend(loc="upper right",handlelength=3)
        plt.ylabel("Flux (W/m$^2$/$\mu$m)",fontsize=fontsize)
        plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()
    out_filename = os.path.join(out_png, "summaryRDIspec_Flambda_bestfit.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))

    plt.show()




    exit()