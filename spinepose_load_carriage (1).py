%pip uninstall mediapipe -y -q
%pip install spinepose opencv-python-headless==4.9.0.80 numpy==1.26.4 matplotlib==3.8.4 openvino-dev -q
print('\nDone. Go to Runtime → Restart session, then run from Cell 2.')

from google.colab import files
import os, cv2, numpy as np, matplotlib, warnings
warnings.filterwarnings('ignore')

# ── Package check ─────────────────────────────────────────────────
print(f'opencv    : {cv2.__version__}')
print(f'numpy     : {np.__version__}')
print(f'matplotlib: {matplotlib.__version__}')
try:
    from spinepose.pose_estimator import SpinePoseEstimator
    print('spinepose : OK')
except Exception as e:
    print(f'spinepose : ERROR — {e}')
    raise

# ── Upload ────────────────────────────────────────────────────────
print('\nSelect BOTH videos now (Ctrl+click / Cmd+click for two files):')
uploaded = files.upload()
keys = list(uploaded.keys())

if len(keys) < 2:
    raise ValueError(f'Need 2 videos, got {len(keys)}: {keys}')

VIDEO_NOLOAD = keys[1]   # ← swap these two lines if order is wrong
VIDEO_LOADED = keys[0]

def vid_info(path):
    cap = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, W, H, total

fps_nl, W_nl, H_nl, tot_nl = vid_info(VIDEO_NOLOAD)
fps_ld, W_ld, H_ld, tot_ld = vid_info(VIDEO_LOADED)

print(f'\n[0] No-load : {VIDEO_NOLOAD}')
print(f'    {W_nl}x{H_nl} px | {fps_nl:.1f} fps | {tot_nl} frames ({tot_nl/fps_nl:.1f}s)')
print(f'\n[1] Loaded  : {VIDEO_LOADED}')
print(f'    {W_ld}x{H_ld} px | {fps_ld:.1f} fps | {tot_ld} frames ({tot_ld/fps_ld:.1f}s)')
print('\nIf the labels above are swapped, edit VIDEO_NOLOAD/VIDEO_LOADED and re-run this cell.')

import warnings, math
warnings.filterwarnings('ignore')
import numpy as np

# ── Subject ───────────────────────────────────────────────────────
SUBJECT_SEX       = 'male'          # 'male' or 'female'
SUBJECT_ID        = 'Subject_01'
SUBJECT_HEIGHT_CM = 183.0           # used for pixel→cm calibration

# ── Output paths ──────────────────────────────────────────────────
OUT_VID_NL    = '/content/annotated_noload.mp4'
OUT_VID_LD    = '/content/annotated_loaded.mp4'
OUT_PLOT_SPINE = '/content/plot_spinal_angles.png'
OUT_PLOT_TRUNK = '/content/plot_trunk_lean.png'
OUT_PLOT_COMP  = '/content/plot_comparison.png'

# ── SpinePose keypoint indices (37-point model) ───────────────────
SPINE_IDX = {
    'C1':36,'C4':35,'C7':17,
    'T3':30,'T8':29,
    'L1':28,'L3':27,'L5':26,
    'Sacrum':19,
}
SPINE_CHAIN = ['C1','C4','C7','T3','T8','L1','L3','L5','Sacrum']

# ── Normative reference values ────────────────────────────────────
# Spinal angles  : Ohlendorf et al. (2023) Scientific Reports
# Trunk Fwd Lean : Lyu & LaBat (2016); Aslam et al. (2025)
#   Normal unloaded walking TFL ~ 4 ± 3 deg forward
#   Load-induced increase documented at ~5.2 ± 1.1 deg for heavy packs
NORMS = {
    'Thoracic Kyphosis': {
        'male':51,'female':56,'std_male':10,'std_female':10,
        'note':'Ohlendorf et al. (2023)'},
    'Lumbar Lordosis': {
        'male':32,'female':49,'std_male':8,'std_female':8,
        'note':'Ohlendorf et al. (2023)'},
    'Lumbar Bending': {
        'male':11,'female':14,'std_male':5,'std_female':5,
        'note':'Ohlendorf et al. (2023)'},
    'Trunk Forward Lean': {
        'male':4,'female':4,'std_male':3,'std_female':3,
        'note':'Lyu & LaBat (2016); Aslam et al. (2025)'},
}

COLORS = {
    'Thoracic Kyphosis' :'#2196F3',
    'Lumbar Lordosis'   :'#FF9800',
    'Lumbar Bending'    :'#4CAF50',
    'Trunk Forward Lean':'#9C27B0',
}

print('Configuration ready.')
print(f'Subject : {SUBJECT_ID} | {SUBJECT_SEX} | {SUBJECT_HEIGHT_CM} cm')
print(f'\nNormative values ({SUBJECT_SEX}):')
for name,n in NORMS.items():
    print(f'  {name:<22} {n[SUBJECT_SEX]:>3} +/- {n[f"std_{SUBJECT_SEX}"]} deg  [{n["note"]}]')

import numpy as np

# Trunk (C7→Sacrum) ≈ 60 % of total standing height — Winter (1990)
TRUNK_FRACTION = 0.60
TRUNK_CM       = SUBJECT_HEIGHT_CM * TRUNK_FRACTION   # = 109.8 cm for 183 cm

def compute_scale(results, label):
    """
    Returns px_per_cm scale from the median C7–Sacrum distance in valid frames.
    Only frames where C7 is above Sacrum (anatomically correct) are used.
    """
    dists = []
    for raw in results:
        if not raw or len(raw) == 0:
            continue
        try:
            fd = np.array(raw[0])
            if fd.ndim == 3: fd = fd[0]
            kpts = fd[:,:2]
            conf = fd[:,2] if fd.shape[1]>2 else np.ones(len(fd))
            C7     = kpts[SPINE_IDX['C7']]
            Sacrum = kpts[SPINE_IDX['Sacrum']]
            if conf[SPINE_IDX['C7']]>0.35 and conf[SPINE_IDX['Sacrum']]>0.35:
                if C7[1] < Sacrum[1]:          # C7 must be above Sacrum in pixel coords
                    dists.append(float(np.linalg.norm(Sacrum - C7)))
        except Exception:
            continue
    if not dists:
        print(f'  [{label}] WARNING: calibration failed — using 1 px/cm fallback')
        return 1.0
    med = float(np.median(dists))
    scale = med / TRUNK_CM
    print(f'  [{label}] C7–Sacrum median: {med:.1f} px | trunk: {TRUNK_CM:.1f} cm | scale: {scale:.3f} px/cm')
    return scale

print(f'Trunk length for calibration: {TRUNK_CM:.1f} cm  ({TRUNK_FRACTION*100:.0f}% of {SUBJECT_HEIGHT_CM} cm)')
print('compute_scale() ready — will be called after inference.')

import cv2, numpy as np, os
from spinepose.pose_estimator import SpinePoseEstimator
from tqdm.notebook import tqdm

os.environ['OPENVINO_NUM_THREADS'] = '1'

print('Initializing SpinePose (YOLOX)...')
estimator = SpinePoseEstimator(detector='yolox', model_version='latest')

def run_inference(video_path, out_path, label):
    cap   = cv2.VideoCapture(video_path)
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    W_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_path, fourcc, fps_v, (W_v, H_v))
    results = []
    pbar = tqdm(total=total, desc=f'[{label}]')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        fr = []
        try:
            bboxes = estimator.detect(frame)
            if bboxes is not None and len(bboxes)>0:
                fr = estimator.estimate(frame,[bboxes[0]])
            results.append(fr)
            if fr and len(fr)>0:
                kd = np.array(fr[0])
                if kd.ndim==3: kd=kd[0]
                pts = kd[:,:2].astype(int)
                chain = [pts[SPINE_IDX[k]] for k in SPINE_CHAIN]
                for j in range(len(chain)-1):
                    cv2.line(frame,tuple(chain[j]),tuple(chain[j+1]),(0,255,0),2)
                for pt in chain:
                    cv2.circle(frame,tuple(pt),4,(0,0,255),-1)
                # Draw C7→Sacrum trunk line in purple
                c7p  = tuple(pts[SPINE_IDX['C7']])
                sacp = tuple(pts[SPINE_IDX['Sacrum']])
                cv2.line(frame,c7p,sacp,(180,0,200),2)
        except Exception:
            results.append([])
        out_vid.write(frame)
        pbar.update(1)
    cap.release(); out_vid.release(); pbar.close()
    print(f'  [{label}] {len(results)} frames | saved → {out_path}')
    return results, fps_v, W_v, H_v

print('\n── No-Load video ──')
results_nl, fps_nl, W_nl, H_nl = run_inference(VIDEO_NOLOAD, OUT_VID_NL, 'No-Load')

print('\n── Loaded video ──')
results_ld, fps_ld, W_ld, H_ld = run_inference(VIDEO_LOADED, OUT_VID_LD, 'Loaded')

print('\n── Calibration ──')
scale_nl = compute_scale(results_nl, 'No-Load')
scale_ld = compute_scale(results_ld, 'Loaded')

print('\nInference complete.')

import numpy as np, math

def cobb_angle(top_pair, bottom_pair):
    """
    Sagittal Cobb angle — inclination-from-vertical method.
    Each segment's tilt from the vertical axis is computed via atan2,
    then the absolute difference gives the divergence angle (= Cobb angle).
    Equivalent to the radiographic perpendicular-lines method.
    Returns 0 if either segment is degenerate.
    """
    v1 = np.array(top_pair[1],    dtype=float) - np.array(top_pair[0],    dtype=float)
    v2 = np.array(bottom_pair[1], dtype=float) - np.array(bottom_pair[0], dtype=float)
    if np.linalg.norm(v1)<1e-9 or np.linalg.norm(v2)<1e-9:
        return 0.0
    a1 = math.degrees(math.atan2(v1[0], abs(v1[1])))
    a2 = math.degrees(math.atan2(v2[0], abs(v2[1])))
    return abs(a1 - a2)

def lateral_deviation_angle(top_pt, mid_pt, bot_pt):
    """
    Lateral bending at mid_pt: supplement of the angle between
    the top→mid and bot→mid vectors (180° = perfectly straight spine).
    Used for coronal-plane lateral bending assessment.
    """
    v1 = np.array(top_pt, dtype=float) - np.array(mid_pt, dtype=float)
    v2 = np.array(bot_pt, dtype=float) - np.array(mid_pt, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-9 or n2<1e-9: return 0.0
    cos_val = np.clip(np.dot(v1,v2)/(n1*n2),-1.0,1.0)
    return 180.0 - math.degrees(math.acos(cos_val))

def trunk_forward_lean(c7, sacrum):
    """
    Sports-2D style Trunk Forward Lean (TFL).
    Angle of the C7→Sacrum vector from the vertical axis.
    Positive = leaning forward (anterior), negative = leaning backward.
    Robust to intermediate keypoint noise because it uses only the
    two most reliably detected endpoints of the trunk.
    Formula: atan2(horizontal_displacement, vertical_displacement)
    """
    v = np.array(sacrum, dtype=float) - np.array(c7, dtype=float)
    if abs(v[1]) < 1e-9: return 0.0
    return math.degrees(math.atan2(v[0], abs(v[1])))

def smooth(arr, k=7):
    if len(arr)<k: return np.array(arr)
    return np.convolve(arr, np.ones(k)/k, mode='same')

def is_valid_frame(kpts, conf):
    """
    Anatomical validity filter:
    — y-coordinates must increase C7 < T3 < T8 < L1 < Sacrum (top→bottom in image)
    — mean confidence of key landmarks must exceed 0.25
    Returns True if the frame passes both checks.
    """
    key_names = ['C7','T3','T8','L1','L3','L5','Sacrum']
    key_idx   = [SPINE_IDX[k] for k in key_names]
    if np.mean([conf[i] for i in key_idx]) < 0.25:
        return False
    C7,T3,T8,L1,Sacrum = [kpts[SPINE_IDX[k]] for k in ['C7','T3','T8','L1','Sacrum']]
    return C7[1] < T3[1] < T8[1] < L1[1] < Sacrum[1]

print('Angle functions defined.')
print('  cobb_angle()              — Cobb sagittal (kyphosis / lordosis)')
print('  lateral_deviation_angle() — coronal bending at L1')
print('  trunk_forward_lean()      — Sports-2D TFL from C7→Sacrum')
print('  is_valid_frame()          — anatomical y-order filter')

import numpy as np

def process_video(results, fps_v, label):
    """
    Extracts per-frame angles from SpinePose results.
    Returns a dict of numpy arrays: time, KY, LO, BE, TFL (all raw).
    """
    time_ax=[]; ky=[]; lo=[]; be=[]; tfl=[]
    skipped=0
    for i,raw in enumerate(results):
        if not raw or len(raw)==0:
            skipped+=1; continue
        try:
            fd = np.array(raw[0])
            if fd.ndim==3: fd=fd[0]
            kpts = fd[:,:2]
            conf = fd[:,2] if fd.shape[1]>2 else np.ones(len(fd))
        except Exception:
            skipped+=1; continue
        if not is_valid_frame(kpts, conf):
            skipped+=1; continue
        C7,T3,T8,L1,L3,L5,Sacrum = [kpts[SPINE_IDX[k]]
            for k in ['C7','T3','T8','L1','L3','L5','Sacrum']]
        time_ax.append(i/fps_v)
        ky.append(cobb_angle([C7,T3],[T8,L1]))
        lo.append(cobb_angle([L1,L3],[L5,Sacrum]))
        be.append(lateral_deviation_angle(C7,L1,Sacrum))
        tfl.append(trunk_forward_lean(C7,Sacrum))
    if len(time_ax)==0:
        raise RuntimeError(f'[{label}] No valid frames. Check video orientation/quality.')
    t   = np.array(time_ax)
    KY  = np.array(ky);  LO = np.array(lo)
    BE  = np.array(be);  TFL= np.array(tfl)
    print(f'[{label}]  valid={len(t)}  skipped={skipped}')
    print(f'  Kyphosis  : {np.mean(KY):.1f} +/- {np.std(KY):.1f} deg')
    print(f'  Lordosis  : {np.mean(LO):.1f} +/- {np.std(LO):.1f} deg')
    print(f'  Bending   : {np.mean(BE):.1f} +/- {np.std(BE):.1f} deg')
    print(f'  TFL       : {np.mean(TFL):.1f} +/- {np.std(TFL):.1f} deg')
    return dict(t=t, KY=KY, LO=LO, BE=BE, TFL=TFL,
                KY_s=smooth(KY), LO_s=smooth(LO),
                BE_s=smooth(BE), TFL_s=smooth(TFL),
                fps=fps_v, label=label)

print('── No-Load ──')
NL = process_video(results_nl, fps_nl, 'No-Load')

print('\n── Loaded ──')
LD = process_video(results_ld, fps_ld, 'Loaded')

# ── Sports-2D style: compute TFL DELTA (loaded - noload baseline) ─
TFL_BASELINE = float(np.mean(NL['TFL']))   # individual unloaded mean
TFL_DELTA    = float(np.mean(LD['TFL'])) - TFL_BASELINE

print(f'\n── Trunk Forward Lean (Sports-2D method) ──')
print(f'  No-load baseline (individual) : {TFL_BASELINE:.2f} deg')
print(f'  Loaded mean                   : {np.mean(LD["TFL"]):.2f} deg')
print(f'  Delta (loaded - baseline)     : {TFL_DELTA:+.2f} deg')
print(f'  Literature expectation (heavy pack): +5.2 +/- 1.1 deg  [Aslam et al. 2025]')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np, warnings
warnings.filterwarnings('ignore')

SPINE_METRICS = ['Thoracic Kyphosis','Lumbar Lordosis','Lumbar Bending']
NL_arrays = [NL['KY'], NL['LO'], NL['BE']]
LD_arrays = [LD['KY'], LD['LO'], LD['BE']]
NL_smooth = [NL['KY_s'],NL['LO_s'],NL['BE_s']]
LD_smooth = [LD['KY_s'],LD['LO_s'],LD['BE_s']]

fig = plt.figure(figsize=(16,18))
fig.patch.set_facecolor('#FAFAFA')
gs  = GridSpec(5,2,figure=fig,hspace=0.55,wspace=0.35)
fig.suptitle(
    f'Spinal Angle Analysis — No-Load vs Loaded (SpinePose CVPR 2025)\n'
    f'{SUBJECT_ID} ({SUBJECT_SEX.title()}) | Calibration: {SUBJECT_HEIGHT_CM} cm',
    fontsize=14,fontweight='bold',y=0.98)

# ── Time-series panels ────────────────────────────────────────────
for row,(name,nl_r,ld_r,nl_s,ld_s) in enumerate(
        zip(SPINE_METRICS,NL_arrays,LD_arrays,NL_smooth,LD_smooth)):
    ax  = fig.add_subplot(gs[row,:])
    col = COLORS[name]
    norm= NORMS[name]
    nval= norm[SUBJECT_SEX]; nstd= norm[f'std_{SUBJECT_SEX}']

    ax.axhspan(nval-nstd,nval+nstd,alpha=0.10,color='green',
               label=f'Normal range ({nval-nstd}–{nval+nstd}°)')
    ax.axhline(nval,color='green',lw=1.5,ls='--',alpha=0.7,
               label=f'Normal mean ({nval}°)')
    # No-load
    ax.plot(NL['t'],nl_r,alpha=0.15,color='steelblue',lw=1)
    ax.plot(NL['t'],nl_s,color='steelblue',lw=2.0,label=f'No-Load ({np.mean(nl_r):.1f}°)')
    # Loaded
    ax.plot(LD['t'],ld_r,alpha=0.15,color='crimson',lw=1)
    ax.plot(LD['t'],ld_s,color='crimson',lw=2.0,ls='--',label=f'Loaded ({np.mean(ld_r):.1f}°)')

    ax.set_ylabel(f'{name} (deg)',fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True,alpha=0.25); ax.set_facecolor('#F5F5F5')
    ax.legend(loc='upper right',fontsize=8,framealpha=0.9)
    ax.text(0.01,0.97,f'Ref: {norm["note"]}',transform=ax.transAxes,
            fontsize=7,va='top',color='gray',style='italic')
    if row==2: ax.set_xlabel('Time (seconds)',fontsize=11)

# ── Bar chart: no-load vs loaded vs normative ─────────────────────
ax_bar = fig.add_subplot(gs[3,0])
short  = ['Kyphosis','Lordosis','Bending']
nl_m   = [float(np.mean(NL[k])) for k in ['KY','LO','BE']]
ld_m   = [float(np.mean(LD[k])) for k in ['KY','LO','BE']]
nm_m   = [NORMS[n][SUBJECT_SEX] for n in SPINE_METRICS]
nm_sd  = [NORMS[n][f'std_{SUBJECT_SEX}'] for n in SPINE_METRICS]
x      = np.arange(3); w=0.25
bars_nl= ax_bar.bar(x-w, nl_m, w, label='No-Load',  color='steelblue',alpha=0.85)
bars_ld= ax_bar.bar(x,   ld_m, w, label='Loaded',   color='crimson',  alpha=0.85)
bars_nm= ax_bar.bar(x+w, nm_m, w, label='Normative',color='gray',     alpha=0.45,
                    yerr=nm_sd, capsize=5)
for b,v in zip(bars_nl,nl_m):
    ax_bar.text(b.get_x()+b.get_width()/2,v+0.3,f'{v:.1f}',ha='center',fontsize=8,color='steelblue',fontweight='bold')
for b,v in zip(bars_ld,ld_m):
    ax_bar.text(b.get_x()+b.get_width()/2,v+0.3,f'{v:.1f}',ha='center',fontsize=8,color='crimson',fontweight='bold')
ax_bar.set_xticks(x); ax_bar.set_xticklabels(short,fontsize=10)
ax_bar.set_ylabel('Angle (deg)',fontsize=10)
ax_bar.set_title('No-Load vs Loaded vs Normative',fontsize=11)
ax_bar.legend(fontsize=8); ax_bar.grid(axis='y',alpha=0.3)
ax_bar.set_facecolor('#F5F5F5')

# ── Violin plot ───────────────────────────────────────────────────
ax_vio = fig.add_subplot(gs[3,1])
positions_nl = [1,3,5]; positions_ld=[1.7,3.7,5.7]
for pos,arr,col in zip(positions_nl,[NL['KY'],NL['LO'],NL['BE']],
                        ['steelblue','steelblue','steelblue']):
    vp=ax_vio.violinplot([arr],positions=[pos],showmeans=True,showmedians=False)
    for pc in vp['bodies']: pc.set_facecolor(col); pc.set_alpha(0.5)
for pos,arr,col in zip(positions_ld,[LD['KY'],LD['LO'],LD['BE']],
                        ['crimson','crimson','crimson']):
    vp=ax_vio.violinplot([arr],positions=[pos],showmeans=True,showmedians=False)
    for pc in vp['bodies']: pc.set_facecolor(col); pc.set_alpha(0.5)
ax_vio.set_xticks([1.35,3.35,5.35])
ax_vio.set_xticklabels(short,fontsize=10)
ax_vio.set_ylabel('Angle (deg)',fontsize=10)
ax_vio.set_title('Angle distribution (blue=No-Load, red=Loaded)',fontsize=10)
ax_vio.grid(axis='y',alpha=0.3); ax_vio.set_facecolor('#F5F5F5')

# ── Summary table ─────────────────────────────────────────────────
ax_tbl = fig.add_subplot(gs[4,:])
ax_tbl.axis('off')
col_labels=['Measurement','No-Load Mean±SD','Loaded Mean±SD',
            'Δ (Loaded−NoLoad)','Normative Mean±SD','Status vs Norm','Reference']
status_colors={'Within normal':'#C8E6C9','Slightly outside':'#FFF9C4','Outside normal':'#FFCDD2'}
rows=[]
for name,nl_arr,ld_arr in zip(SPINE_METRICS,
    [NL['KY'],NL['LO'],NL['BE']],[LD['KY'],LD['LO'],LD['BE']]):
    norm=NORMS[name]
    nval=norm[SUBJECT_SEX]; nstd=norm[f'std_{SUBJECT_SEX}']
    nl_mean=float(np.mean(nl_arr)); nl_std=float(np.std(nl_arr))
    ld_mean=float(np.mean(ld_arr)); ld_std=float(np.std(ld_arr))
    delta=ld_mean-nl_mean
    z=(ld_mean-nval)/nstd
    if abs(z)<=1.0: status='Within normal'
    elif abs(z)<=2.0: status='Slightly outside'
    else: status='Outside normal'
    rows.append([name,
        f'{nl_mean:.1f}±{nl_std:.1f}°',f'{ld_mean:.1f}±{ld_std:.1f}°',
        f'{delta:+.1f}°',f'{nval}±{nstd}°',status,norm['note']])
tbl=ax_tbl.table(cellText=rows,colLabels=col_labels,loc='center',cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,2.1)
for j in range(len(col_labels)):
    tbl[0,j].set_facecolor('#37474F')
    tbl[0,j].set_text_props(color='white',fontweight='bold')
for i,row in enumerate(rows):
    for j in range(len(col_labels)):
        tbl[i+1,j].set_facecolor(
            status_colors.get(row[5],'white') if j==5
            else ('#ECEFF1' if i%2==0 else 'white'))
ax_tbl.set_title('Spinal Angles — Summary vs Ohlendorf et al. (2023)',
                  fontsize=11,pad=10,fontweight='bold')

plt.savefig(OUT_PLOT_SPINE,dpi=150,bbox_inches='tight',facecolor='#FAFAFA')
plt.show()
print(f'Saved → {OUT_PLOT_SPINE}')

import matplotlib.pyplot as plt
import numpy as np, warnings
warnings.filterwarnings('ignore')

tfl_nl = NL['TFL']; tfl_nl_s = NL['TFL_s']; t_nl = NL['t']
tfl_ld = LD['TFL']; tfl_ld_s = LD['TFL_s']; t_ld = LD['t']

mean_nl  = float(np.mean(tfl_nl))
mean_ld  = float(np.mean(tfl_ld))
delta    = mean_ld - mean_nl
std_nl   = float(np.std(tfl_nl))
std_ld   = float(np.std(tfl_ld))

fig, axes = plt.subplots(1,3,figsize=(18,5))
fig.patch.set_facecolor('#FAFAFA')
fig.suptitle(
    f'Trunk Forward Lean — Sports-2D Baseline Method\n'
    f'{SUBJECT_ID} ({SUBJECT_SEX.title()}) | Baseline = individual no-load mean',
    fontsize=13,fontweight='bold')

# Panel A — raw TFL over time, both conditions
ax=axes[0]
ax.plot(t_nl,tfl_nl,alpha=0.2,color='steelblue',lw=1)
ax.plot(t_nl,tfl_nl_s,color='steelblue',lw=2.2,label=f'No-Load ({mean_nl:.1f}°)')
ax.plot(t_ld,tfl_ld,alpha=0.2,color='crimson',lw=1)
ax.plot(t_ld,tfl_ld_s,color='crimson',lw=2.2,ls='--',label=f'Loaded ({mean_ld:.1f}°)')
ax.axhline(mean_nl,color='steelblue',lw=1,ls=':',alpha=0.8)
ax.axhline(mean_ld,color='crimson',  lw=1,ls=':',alpha=0.8)
ax.axhspan(NORMS['Trunk Forward Lean'][SUBJECT_SEX]-NORMS['Trunk Forward Lean'][f'std_{SUBJECT_SEX}'],
           NORMS['Trunk Forward Lean'][SUBJECT_SEX]+NORMS['Trunk Forward Lean'][f'std_{SUBJECT_SEX}'],
           alpha=0.10,color='green',label='Population normal range')
ax.set_xlabel('Time (s)',fontsize=11); ax.set_ylabel('TFL (deg)',fontsize=11)
ax.set_title('TFL over time',fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_facecolor('#F5F5F5')

# Panel B — delta TFL (relative to individual baseline)
ax=axes[1]
delta_series = tfl_ld - mean_nl   # frame-by-frame delta from no-load baseline
delta_s      = smooth(delta_series)
ax.axhline(0,color='steelblue',lw=1.5,ls='--',label='No-load baseline (0°)')
ax.plot(t_ld,delta_series,alpha=0.2,color='crimson',lw=1)
ax.plot(t_ld,delta_s,color='crimson',lw=2.2,label=f'Δ TFL loaded ({delta:+.1f}°)')
ax.axhline(5.2,color='orange',lw=1.5,ls=':',
           label='Expected Δ heavy pack +5.2° [Aslam 2025]')
ax.fill_between(t_ld,delta_s-std_ld,delta_s+std_ld,alpha=0.1,color='crimson')
ax.set_xlabel('Time (s)',fontsize=11); ax.set_ylabel('Δ TFL from baseline (deg)',fontsize=11)
ax.set_title('Δ Trunk Lean (Loaded − No-Load baseline)',fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_facecolor('#F5F5F5')

# Panel C — bar + individual data points
ax=axes[2]
bars=ax.bar(['No-Load','Loaded'],[mean_nl,mean_ld],
            color=['steelblue','crimson'],alpha=0.75,
            yerr=[std_nl,std_ld],capsize=8,
            error_kw={'linewidth':2})
ax.scatter(['No-Load']*len(tfl_nl),tfl_nl,color='steelblue',alpha=0.3,s=8,zorder=3)
ax.scatter(['Loaded']*len(tfl_ld), tfl_ld, color='crimson',  alpha=0.3,s=8,zorder=3)
for b,v in zip(bars,[mean_nl,mean_ld]):
    ax.text(b.get_x()+b.get_width()/2,v+std_nl+0.3,f'{v:.1f}°',
            ha='center',fontsize=11,fontweight='bold')
ax.axhline(NORMS['Trunk Forward Lean'][SUBJECT_SEX],color='green',
           lw=1.5,ls='--',label='Population mean (4°)')
ax.set_ylabel('TFL (deg)',fontsize=11)
ax.set_title(f'Mean TFL: Δ = {delta:+.1f}° from individual baseline',fontsize=11)
ax.legend(fontsize=9); ax.grid(axis='y',alpha=0.3); ax.set_facecolor('#F5F5F5')
ax.text(0.5,-0.12,
    f'Baseline method: no-load individual mean ({mean_nl:.1f}°) used as reference\n'
    f'Sports-2D approach — Lyu & LaBat (2016); Aslam et al. (2025)',
    transform=ax.transAxes,ha='center',fontsize=8,color='gray',style='italic')

plt.tight_layout()
plt.savefig(OUT_PLOT_TRUNK,dpi=150,bbox_inches='tight',facecolor='#FAFAFA')
plt.show()
print(f'Saved → {OUT_PLOT_TRUNK}')

import numpy as np

sep = '='*72
print(sep)
print('  SPINAL POSTURE & TRUNK LEAN ANALYSIS REPORT')
print('  SpinePose (CVPR 2025) + Sports-2D Trunk Forward Lean method')
print(sep)
print(f'  Subject    : {SUBJECT_ID} ({SUBJECT_SEX.title()})')
print(f'  Height     : {SUBJECT_HEIGHT_CM} cm | Trunk = {TRUNK_CM:.1f} cm (60% of height)')
print(f'  Scale (NL) : {scale_nl:.4f} px/cm  |  Scale (LD): {scale_ld:.4f} px/cm')
print(f'  No-Load    : {len(NL["t"])} valid frames ({NL["t"][-1]:.2f}s @ {fps_nl:.0f}fps)')
print(f'  Loaded     : {len(LD["t"])} valid frames ({LD["t"][-1]:.2f}s @ {fps_ld:.0f}fps)')
print(sep)
print(f'  {"TRUNK FORWARD LEAN (Sports-2D baseline method)":<50}')
print(f'  {"─"*68}')
print(f'  No-Load baseline (individual mean) : {float(np.mean(NL["TFL"])):>7.2f} deg')
print(f'  Loaded mean                        : {float(np.mean(LD["TFL"])):>7.2f} deg')
print(f'  Delta (loaded − baseline)          : {TFL_DELTA:>+7.2f} deg')
print(f'  Population normal (Lyu 2016)       : {"4 +/- 3 deg":>10}')
print(f'  Expected delta heavy pack (Aslam)  : {"+5.2 +/- 1.1 deg":>18}')
print(sep)
print(f'  {"SPINAL ANGLES (Cobb sagittal method)":<50}')
print(f'  {"─"*68}')
print(f'  {"Measurement":<22} {"No-Load":>12} {"Loaded":>12} {"Δ":>8} {"Norm":>10}  Status')
print(f'  {"─"*68}')
for name,nk,lk in zip(["Thoracic Kyphosis","Lumbar Lordosis","Lumbar Bending"],
                       ["KY","LO","BE"],["KY","LO","BE"]):
    norm=NORMS[name]; nval=norm[SUBJECT_SEX]
    nl_m=float(np.mean(NL[nk])); ld_m=float(np.mean(LD[lk]))
    delta_s=ld_m-nl_m
    z=(ld_m-nval)/norm[f'std_{SUBJECT_SEX}']
    flag='Within normal' if abs(z)<=1 else ('Slightly outside' if abs(z)<=2 else 'Outside normal')
    print(f'  {name:<22} {nl_m:>10.1f}°  {ld_m:>10.1f}°  {delta_s:>+6.1f}°  {nval:>8}°  {flag}')
print(sep)
print('  METHODOLOGY (for Methods section)')
print(f'  {"─"*68}')
print('  Vertebral keypoints (C1,C4,C7,T3,T8,L1,L3,L5,Sacrum) detected by')
print('  SpinePose (CVPR 2025), a 37-keypoint spine-specific model trained on')
print('  SpineTrack (58,000+ annotated images). Frames with anatomically')
print('  invalid keypoint ordering (y-axis violation) were excluded.')
print()
print('  SPINAL ANGLES: Thoracic kyphosis = Cobb sagittal angle between C7→T3')
print('  and T8→L1 segments (inclination-from-vertical method). Lumbar lordosis')
print('  = Cobb sagittal angle between L1→L3 and L5→Sacrum. Lateral bending =')
print('  deviation angle at L1 between C7→L1 and L1→Sacrum vectors.')
print()
print('  TRUNK FORWARD LEAN: C7→Sacrum vector angle from vertical (atan2 method)')
print(f'  calibrated to subject height ({SUBJECT_HEIGHT_CM} cm; trunk = {TRUNK_CM:.1f} cm,')
print('  60% of height per Winter 1990). The no-load walking condition serves')
print('  as the individual baseline, mirroring the Sports-2D approach. The')
print('  loaded delta quantifies additional forward lean induced by load carriage.')
print()
print('  REFERENCES')
print('  [1] Lyu & LaBat (2016). Int J Ind Ergonomics 56:115-123.')
print('  [2] Aslam et al. (2025). INSIGHTS-JOURNAL 3(3):454-460.')
print('  [3] Ohlendorf et al. (2023). Scientific Reports 13:12395.')
print('  [4] Winter DA (1990). Biomechanics of Human Movement. Wiley.')
print('  [5] SpinePose/SpineTrack (CVPR 2025).')
print(sep)

from google.colab import files
import os

outputs = {
    OUT_VID_NL    : 'Annotated no-load video (MP4)',
    OUT_VID_LD    : 'Annotated loaded video (MP4)',
    OUT_PLOT_SPINE: 'Spinal angles figure (PNG)',
    OUT_PLOT_TRUNK: 'Trunk forward lean figure (PNG)',
}

for path,desc in outputs.items():
    if os.path.exists(path):
        sz = os.path.getsize(path)/1024/1024
        print(f'Downloading: {path}  [{desc}]  {sz:.1f} MB')
        files.download(path)
    else:
        print(f'NOT FOUND: {path} — make sure all previous cells ran successfully.')