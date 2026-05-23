import os, glob, zipfile, time, joblib, shutil
import numpy as np
import rasterio
import rasterio.warp
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.ndimage import gaussian_filter

# ----------------- CONFIG -----------------
DRONE_FOLDER = r"C:\Septem20_drone"
S2_TIF = r"C:\september_20_sentinel\KVK_20_09_24.tif"
PS_ZIP = r"C:\Spetember20_PS\September_20_psscene_analytic_8b_sr_udm2.zip"
TMP_EX = "temp_ps_extraction"
OUT_DIR = r"C:\output_try"
os.makedirs(OUT_DIR, exist_ok=True)

REQUIRED_BANDS = ['RED','GREEN','NIR','RED_EDGE']
S2_SCALE = 10000.0
PS_SCALE = 10000.0
NODATA = -9999.0
RANDOM = 42
MAX_SAMPLES = 1_000_000

# rasterio uses 1-based band indices
S2_BANDS = {'GREEN':3, 'RED':4, 'RED_EDGE':5, 'NIR':8}
PS_BANDS = {'GREEN':3, 'RED':5, 'RED_EDGE':6, 'NIR':7}  # adjust if your PS stack differs

MODEL_CHOICE = 'rf'   # 'rf' or 'gbr'
GAUSS_SIGMA = 0.0     # smoothing on calibrated bands (0 to disable)
# ------------------------------------------

def unzip_ps_find_stack(zip_path, tmp_dir):
    outdir = os.path.join(os.path.dirname(zip_path), tmp_dir)
    os.makedirs(outdir, exist_ok=True)
    with zipfile.ZipFile(zip_path,'r') as z:
        z.extractall(outdir)
    tifs = glob.glob(os.path.join(outdir,'**','*.tif'), recursive=True)
    # try to find analyticms stack by name; fallback to largest multiband
    stack = next((f for f in tifs if 'analyticms' in os.path.basename(f).lower()), None)
    if stack is None:
        mult = [f for f in tifs if rasterio.open(f).count>1]
        stack = max(mult, key=lambda p: os.path.getsize(p)) if mult else None
    return stack, outdir

def find_drone_bands(folder):
    files = glob.glob(os.path.join(folder,'*.tif')) + glob.glob(os.path.join(folder,'*.tiff'))
    mapping = {}
    keys = {'red':'RED','green':'GREEN','nir':'NIR','red_edge':'RED_EDGE','rededge':'RED_EDGE','red edge':'RED_EDGE'}
    for f in files:
        nm = os.path.basename(f).lower()
        for k,v in keys.items():
            if k in nm and v not in mapping:
                mapping[v]=f
    if len(mapping)!=4:
        raise RuntimeError(f"Found drone bands: {mapping.keys()}; need all 4.")
    return mapping

def resample_to_profile(src_path, band_index, dst_profile):
    with rasterio.open(src_path) as src:
        dst = np.empty((dst_profile['height'], dst_profile['width']), dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, band_index),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_profile['transform'], dst_crs=dst_profile['crs'],
            resampling=rasterio.warp.Resampling.bilinear
        )
    return dst

def prepare_training(drone_map, s2_tif, ps_stack):
    # read drone reference profile from RED band
    with rasterio.open(drone_map['RED']) as dr:
        prof = dr.profile.copy()
        h,w = prof['height'], prof['width']

    # read drone bands into arrays and build valid mask
    drone = {}
    valid = np.ones((h,w), dtype=bool)
    for b in REQUIRED_BANDS:
        with rasterio.open(drone_map[b]) as ds:
            arr = ds.read(1).astype(np.float32)
            mask = (arr!=ds.nodata) if ds.nodata is not None else np.isfinite(arr)
            drone[b]=arr
            valid &= mask

    # resample PS to drone grid
    ps_res = {}
    stack = ps_stack
    with rasterio.open(stack) as src:
        for bname, idx in PS_BANDS.items():
            arr = np.empty((h,w), dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(src, idx),
                destination=arr,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=prof['transform'], dst_crs=prof['crs'],
                resampling=rasterio.warp.Resampling.bilinear
            )
            ps_res[bname] = arr / PS_SCALE
            valid &= np.isfinite(arr)

    # resample S2 to drone grid
    s2_res = {}
    with rasterio.open(s2_tif) as src:
        for bname, idx in S2_BANDS.items():
            arr = src.read(idx).astype(np.float32) / S2_SCALE
            out = np.empty((h,w), dtype=np.float32)
            rasterio.warp.reproject(
                source=arr, destination=out,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=prof['transform'], dst_crs=prof['crs'],
                resampling=rasterio.warp.Resampling.bilinear
            )
            s2_res[bname]=out
            valid &= np.isfinite(out)

    # final sample indices
    flat_valid = valid.flatten()
    idxs = np.where(flat_valid)[0]
    if idxs.size==0:
        raise RuntimeError("No overlapping valid pixels.")
    if idxs.size > MAX_SAMPLES:
        np.random.seed(RANDOM); idxs = np.random.choice(idxs, size=MAX_SAMPLES, replace=False)

    # build X (PS + S2) and Y (drone)
    X_list=[]; Y_list=[]
    for b in REQUIRED_BANDS:
        X_list.append(ps_res[b].flatten()[idxs])
    for b in REQUIRED_BANDS:
        X_list.append(s2_res[b].flatten()[idxs])
    for b in REQUIRED_BANDS:
        Y_list.append(drone[b].flatten()[idxs].astype(np.float32))

    X = np.column_stack(X_list)
    Y = np.column_stack(Y_list)
    return X, Y, prof, stack

def train_model(X,Y):
    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X = X[mask]; Y = Y[mask]
    Xtr,Xte,Ytr,Yte = train_test_split(X,Y,test_size=0.2,random_state=RANDOM)
    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    if MODEL_CHOICE=='rf':
        base = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM)
    else:
        base = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=RANDOM)
    model = MultiOutputRegressor(base)
    t0=time.time(); model.fit(Xtr_s, Ytr); print("Train time",time.time()-t0)
    Yp = model.predict(Xte_s)
    per_band = {}
    for i,b in enumerate(REQUIRED_BANDS):
        r2 = r2_score(Yte[:,i], Yp[:,i]); rmse = np.sqrt(mean_squared_error(Yte[:,i], Yp[:,i]))
        per_band[b] = {'r2':r2,'rmse':rmse}
    # save artifacts
    joblib.dump(model, os.path.join(OUT_DIR,'ps2dr_model.joblib'))
    joblib.dump(scaler, os.path.join(OUT_DIR,'ps_scaler.joblib'))
    print("Saved model and scaler. Metrics:", per_band)
    return model, scaler

def infer_and_write(model, scaler, ps_stack, s2_tif, outdir, profile_ps_ref=None):
    # open PS stack to get PS grid/profile
    with rasterio.open(ps_stack) as ps_src:
        ps_prof = ps_src.profile
        h,w = ps_prof['height'], ps_prof['width']

        # build X_inference (PS bands + S2 warped to PS grid)
        Xinf = np.zeros((h*w, len(REQUIRED_BANDS)*2), dtype=np.float32)
        for i,b in enumerate(REQUIRED_BANDS):
            arr = ps_src.read(PS_BANDS[b]).astype(np.float32) / PS_SCALE
            Xinf[:, i] = arr.flatten()
        # warp each S2 band to PS grid
        with rasterio.open(s2_tif) as s2:
            for i,b in enumerate(REQUIRED_BANDS):
                s2_arr = s2.read(S2_BANDS[b]).astype(np.float32) / S2_SCALE
                dst = np.empty((h,w), dtype=np.float32)
                rasterio.warp.reproject(
                    source=s2_arr, destination=dst,
                    src_transform=s2.transform, src_crs=s2.crs,
                    dst_transform=ps_src.transform, dst_crs=ps_src.crs,
                    resampling=rasterio.warp.Resampling.bilinear
                )
                Xinf[:, i+4] = dst.flatten()

        # mask valid PS pixels using PS band1 nodata
        mask_ps = ~ps_src.read(1, masked=True).mask.flatten()
    # apply scaler + model in batches
    scaler_exists = scaler is not None
    model_loaded = model
    Ycal = np.full((h*w, 4), np.nan, dtype=np.float32)
    ensemble_var = np.full((h*w,), np.nan, dtype=np.float32)  # variance across ensemble if RF
    batch=200_000
    for s in range(0, mask_ps.sum(), batch):
        valid_idx = np.where(mask_ps)[0][s:s+batch]
        Xb = Xinf[valid_idx,:]
        Xb_s = scaler.transform(Xb) if scaler_exists else Xb
        preds = model_loaded.predict(Xb_s)  # shape (n,4)
        Ycal[valid_idx,:] = preds
        # approximate confidence: if base estimator attribute exists compute per-sample variance
        try:
            # only works for RF base estimator within MultiOutputRegressor
            import numpy as _np
            # collect predictions per base estimator if MultiOutputRegressor with RF estimators
            ecs = model_loaded.estimators_  # list of estimators (one per target), if multioutput -> complicated
            # fallback: we won't compute per-pixel variance here (skip)
        except Exception:
            pass

    # optional smoothing
    if GAUSS_SIGMA>0:
        for b in range(4):
            arr2 = Ycal[:,b].reshape(h,w)
            nanmask = ~np.isfinite(arr2)
            arr_fill = arr2.copy()
            arr_fill[nanmask]=np.nanmean(arr_fill)
            arr_sm = gaussian_filter(arr_fill, sigma=GAUSS_SIGMA)
            arr_sm[nanmask]=np.nan
            Ycal[:,b] = arr_sm.flatten()

    # write calibrated bands as GeoTIFFs (order = RED,GREEN,RED_EDGE,NIR)
    out_profile = ps_prof.copy()
    out_profile.update(count=1,dtype=rasterio.float32, nodata=NODATA)
    band_order = ['RED','GREEN','RED_EDGE','NIR']
    for bi,b in enumerate(band_order):
        arr = Ycal[:, bi].reshape(h,w)
        outpath = os.path.join(outdir, f'Calibrated_PS_{b}.tif')
        with rasterio.open(outpath,'w',**out_profile) as dst:
            dst.write(np.where(np.isfinite(arr), arr, NODATA).astype(np.float32),1)
        print("Wrote", outpath)

    # compute indices and write them
    red = Ycal[:,0]; green = Ycal[:,1]; re = Ycal[:,2]; nir = Ycal[:,3]
    valid_idx = np.isfinite(red)&np.isfinite(nir)
    ndvi = np.full(h*w, np.nan); ndvi[valid_idx] = (nir[valid_idx]-red[valid_idx])/(nir[valid_idx]+red[valid_idx]+1e-8)
    ndre = np.full(h*w, np.nan); ndre[valid_idx] = (nir[valid_idx]-re[valid_idx])/(nir[valid_idx]+re[valid_idx]+1e-8)
    gndvi = np.full(h*w, np.nan); gndvi[valid_idx] = (nir[valid_idx]-green[valid_idx])/(nir[valid_idx]+green[valid_idx]+1e-8)
    rvi = np.full(h*w, np.nan); rvi[valid_idx] = nir[valid_idx]/(red[valid_idx]+1e-8)
    idxs = {'NDVI':ndvi,'NDRE':ndre,'GNDVI':gndvi,'RVI':rvi}
    for name, arr in idxs.items():
        outpath = os.path.join(outdir, f'Calibrated_PS_{name}.tif')
        with rasterio.open(outpath,'w',**out_profile) as dst:
            dst.write(np.where(np.isfinite(arr), arr, NODATA).astype(np.float32),1)
        print("Wrote", outpath)

    # save flattened valid indices to NP/CSV
    valid_rows = np.where(np.isfinite(ndvi))[0]
    flat_indices = np.vstack([ndvi[valid_rows], ndre[valid_rows], gndvi[valid_rows], rvi[valid_rows]]).T
    np.save(os.path.join(outdir,'Calibrated_PS_Indices_FLAT.npy'), flat_indices)
    np.savetxt(os.path.join(outdir,'Calibrated_PS_Indices_FLAT.csv'), flat_indices, delimiter=',', header=','.join(['NDVI','NDRE','GNDVI','RVI']), comments='')
    print("Saved flat indices.")

# ---------- main ----------
if __name__=='__main__':
    ps_stack, tmpdir = unzip_ps_find_stack(PS_ZIP, TMP_EX)
    drone = find_drone_bands(DRONE_FOLDER)
    X,Y, drone_prof, _ = prepare_training(drone, S2_TIF, ps_stack)
    model, scaler = None, None
    model, scaler = train_model(X,Y)
    infer_and_write(model, scaler, ps_stack, S2_TIF, OUT_DIR)
    # cleanup
    try:
        shutil.rmtree(tmpdir)
    except: pass
    print("Done. Outputs in", OUT_DIR)
