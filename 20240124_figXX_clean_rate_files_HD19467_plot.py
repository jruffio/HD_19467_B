import os
import time
import jwst

# Standard imports
import numpy as np
from astropy.io import fits
from glob import glob

from copy import copy
from scipy.stats import median_abs_deviation
from breads.fit import fitfm

# Print out what pipeline version we're using
print('JWST pipeline version',jwst.__version__)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patheffects as PathEffects

fontsize = 12
# output dir for images
out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
# out_png = out_dir.replace("xy","figures")
if not os.path.exists(out_png):
    os.makedirs(out_png)
targetdir = "/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/"
det1_dir = os.path.join(targetdir,"20240124_stage1") # Detector1 pipeline outputs will go here
clean_dir = os.path.join(targetdir,"20240220_stage1_clean_test") # Detector1 pipeline outputs will go here
rate_files = glob(os.path.join(det1_dir,"jw*_rate.fits"))
rate_files.sort()
rate_file = rate_files[0]
clean_rate_file = os.path.join(clean_dir,os.path.basename(rate_file))

from breads.utils import get_spline_model
def fm_column_background(nonlin_paras, cubeobj, nodes=20, fix_parameters=None,return_where_finite=False,fit_diffus=False,
             regularization=None,badpixfraction=0.75, M_spline=None):
    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    if M_spline is None:
        if type(nodes) is int:
            N_nodes = nodes
            x_knots = np.linspace(0, np.size(cubeobj.data), N_nodes, endpoint=True).tolist()
        elif type(nodes) is list or type(nodes) is np.ndarray:
            x_knots = nodes
            if type(nodes[0]) is list or type(nodes[0]) is np.ndarray:
                N_nodes = np.sum([np.size(n) for n in nodes])
            else:
                N_nodes = np.size(nodes)
        else:
            raise ValueError("Unknown format for nodes.")
    else:
        N_nodes = M_spline.shape[1]

    # Number of linear parameters
    N_linpara = N_nodes
    if fit_diffus:
        N_linpara += 1

    data = cubeobj.data
    noise = cubeobj.noise
    bad_pixels = cubeobj.bad_pixels

    where_trace_finite = np.where(np.isfinite(data)*np.isfinite(bad_pixels)*(noise!=0))
    d = data[where_trace_finite]
    s = noise[where_trace_finite]

    # print("coucou")
    # print(np.size(where_trace_finite[0]), (1-badpixfraction) * np.sum(new_mask), vsini < 0)
    if np.size(where_trace_finite[0]) <= (1-badpixfraction) * np.size(data):
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        x = np.arange(np.size(cubeobj.data))
        if M_spline is None:
            M = get_spline_model(x_knots, x, spline_degree=3)
        else:
            M =copy(M_spline)
        # print(M_spline.shape)

        if fit_diffus:
            diffus_center = _nonlin_paras[0]
            diffus_scale = _nonlin_paras[1]
            diffus_model = np.exp(-np.abs(x-diffus_center)/diffus_scale)
            # diffus_model = np.exp(-(x-diffus_center)**2/diffus_scale**2)
            # diffus_model = 1-np.abs(x-diffus_center)/diffus_scale
            M_diffus = diffus_model[:,None]
            M = np.concatenate([M_diffus,M], axis=1)

        M = M[where_trace_finite[0],:]

        extra_outputs = {}
        if regularization == "default":
            s_reg = np.array(np.nanmax(d) + np.zeros(N_nodes))
            d_reg = np.array(0 + np.zeros(N_nodes))
            if fit_diffus:
                s_reg = np.concatenate([[np.nan],s_reg])
                d_reg = np.concatenate([[np.nan],d_reg])
            extra_outputs["regularization"] = (d_reg, s_reg)
        elif regularization == "user":
            raise Exception("user defined regularisation not yet implemented")
            extra_outputs["regularization"] = (d_reg, s_reg)

        if return_where_finite:
            extra_outputs["where_trace_finite"] = where_trace_finite

        if len(extra_outputs) >= 1:
            return d, M, s, extra_outputs
        else:
            return d, M, s



if 1:
    N_nodes = 40
    x = np.arange(2048)
    x_knots = np.linspace(0, 2048, N_nodes, endpoint=True).tolist()
    M_spline = get_spline_model(x_knots, x, spline_degree=3)

    basename = os.path.basename(rate_file)
    cal_filename = os.path.join(targetdir,"20240124_stage2/"+basename.replace("_rate.fits","_cal.fits"))
    print(cal_filename)
    print(glob(cal_filename))
    with fits.open(cal_filename) as hdul:
        cal_im = hdul["SCI"].data

    # Get data
    hdul = fits.open(rate_file)

    if 1: #JB clean
        priheader = hdul[0].header
        extheader = hdul[1].header
        im = hdul["SCI"].data
        im_ori = copy(im)
        noise = hdul["ERR"].data
        dq = hdul["DQ"].data
        ny, nx = im.shape

        from breads.instruments.jwstnirspec_cal import untangle_dq
        # Simplifying bad pixel map following convention in this package as: nan = bad, 1 = good
        # bad_pixels = np.ones((ny, nx))
        bad_pixels=np.zeros(cal_im.shape)+np.nan
        bad_pixels[np.where(np.isnan(cal_im))] = 1
        # Pixels marked as "do not use" are marked as bad (nan = bad, 1 = good):
        bad_pixels[np.where(untangle_dq(dq, verbose=True)[0, :, :])] = np.nan
        bad_pixels[np.where(np.isnan(im))] = np.nan
        im[np.where(np.isnan(im))] = 0

        #Removing any data with zero noise
        where_zero_noise = np.where(noise == 0)
        noise[where_zero_noise] = np.nan
        bad_pixels[where_zero_noise] = np.nan

        if "nrs1" in rate_file:
            for rowid in range(im.shape[0]):
                finite_ids = np.where(np.isfinite(cal_im[rowid,0:450]))[0]
                if len(finite_ids) != 0 :
                    id_to_mask = np.min(finite_ids)
                    bad_pixels[rowid,0:id_to_mask] = np.nan
        elif "nrs2" in rate_file:
            for rowid in range(im.shape[0]):
                finite_ids = np.where(np.isfinite(cal_im[rowid,1550::]))[0]
                if len(finite_ids) != 0 :
                    id_to_mask = np.max(finite_ids)
                    bad_pixels[rowid,1550+id_to_mask::] = np.nan
        bad_pixels[np.where(np.abs(im)>5)] = np.nan

        from breads.instruments.instrument import Instrument
        data = Instrument()
        new_im = np.zeros(im.shape)
        if 1:
            colid = 350
            data.data = copy(im[:,colid])
            data.noise = copy(noise[:,colid])
            data.bad_pixels = copy(bad_pixels[:,colid])

            fm_paras = {"badpixfraction":0.99,"nodes":N_nodes,"fix_parameters": [],"regularization":"default","fit_diffus":False,"M_spline":M_spline}
            # fm_paras = {"badpixfraction":0.99,"nodes":N_nodes,"fix_parameters": nonlin_paras,"regularization":"default","fit_diffus":True}
            log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm([],data,fm_column_background,fm_paras)

            d_masked, M, s,extra_outputs = fm_column_background([],data,return_where_finite=True,**fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            data.bad_pixels = np.ones(data.data.shape)
            d, M, s,_ = fm_column_background([],data,return_where_finite=True,**fm_paras)
            # print(data.data.shape,d.shape,np.size(where_finite[0]),np.size(d_masked))
            d_masked_canvas = np.zeros(d.shape)+np.nan
            d_masked_canvas[where_finite] = d_masked

            m = np.dot(M,linparas)
            new_im[:,colid] = im_ori[:,colid]-m

            data.bad_pixels = bad_pixels[:,colid]
            mad = median_abs_deviation(((d_masked_canvas-m))[np.where(np.isfinite(d_masked_canvas))])
            data.bad_pixels[np.where(np.abs(d_masked_canvas-m)>5*mad)] = np.nan
            # plt.plot(data.bad_pixels)
            # plt.show()

            # Redo the fit with outliers removed
            log_prob, log_prob_H0, rchi2, linparas, linparas_err = fitfm([],data,fm_column_background,fm_paras)
            d_masked, M, s,extra_outputs = fm_column_background([],data,return_where_finite=True,**fm_paras)
            where_finite = extra_outputs["where_trace_finite"]
            data.bad_pixels = np.ones(data.data.shape)
            d, M, s,_ = fm_column_background([],data,return_where_finite=True,**fm_paras)
            # print(data.data.shape,d.shape,np.size(where_finite[0]),np.size(d_masked))
            d_masked_canvas = np.zeros(d.shape)+np.nan
            d_masked_canvas[where_finite] = d_masked

            m = np.dot(M,linparas)
            new_im[:,colid] = im_ori[:,colid]-m

            color_list = ["#ff9900", "#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
            plt.figure(1,figsize=(12,6))
            plt.subplot(2,1,1)
            plt.plot(d,label="Data",color=color_list[0],linewidth=1)
            plt.plot(d_masked_canvas,label="Background pixels",color=color_list[1],linewidth=2)
            # plt.plot(m,label="model",linestyle="--")
            plt.ylim([-5,200])
            # plt.xlabel("Detector row index (pixel)",fontsize=fontsize)
            plt.ylabel(r"Flux (DN/s)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.legend(loc="upper left", handlelength=3,fontsize=fontsize)

            plt.subplot(2,1,2)
            # plt.plot(d,label="Data",color=color_list[0])
            plt.plot(d_masked_canvas,label="Background pixels",linewidth=2,color=color_list[1])
            plt.plot(m,label="Best fit model",linestyle="--",linewidth=1,color=color_list[2])
            plt.plot(d - m,label="Corrected data",linestyle="-",color= color_list[0],linewidth=1)
            plt.ylim([-0.5,2])

            plt.xlabel("Detector row index (pixel)",fontsize=fontsize)
            plt.ylabel(r"Flux (DN/s)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
            plt.legend(loc="upper left", handlelength=3,fontsize=fontsize)

            plt.tight_layout()

            out_filename = os.path.join(out_png, "rate_clean_HD19467.png")
            print("Saving " + out_filename)
            plt.savefig(out_filename, dpi=300)
            plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')

            plt.figure(2,figsize=(12,4.5))
            hdul = fits.open(clean_rate_file)
            cleaned_im = hdul["SCI"].data
            cleaned_im[np.where(np.isnan(cleaned_im))]=0
            plt.subplot(1,2,1)
            plt.imshow(im,origin="lower")
            # plt.imshow(im,origin="lower",aspect="auto")
            plt.clim([-0.1,0.2])
            txt = plt.text(0.03, 0.99, "Before", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="white")
            # txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
            plt.xticks([])
            plt.yticks([])
            cbar = plt.colorbar(pad=0)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(label='DN/s', fontsize=fontsize)

            plt.subplot(1,2,2)
            plt.imshow(cleaned_im,origin="lower")
            # plt.imshow(cleaned_im,origin="lower",aspect="auto")
            plt.clim([-0.1,0.2])
            txt = plt.text(0.03, 0.99, "After", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="white")
            # txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
            plt.xticks([])
            plt.yticks([])
            cbar = plt.colorbar(pad=0)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(label='DN/s', fontsize=fontsize)


            out_filename = os.path.join(out_png, "rate_clean_HD19467_images.png")
            print("Saving " + out_filename)
            plt.savefig(out_filename, dpi=300)
            plt.savefig(out_filename.replace(".png", ".pdf"),bbox_inches='tight')

            plt.show()

exit()