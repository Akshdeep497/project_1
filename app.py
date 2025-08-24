# Streamlit: Loitering + Abandoned Bag + Robust Unusual (running) detection
# Less sensitive unusual: camera stabilization + body-lengths/sec + longer sustain
# Run: pip install -r requirements.txt && streamlit run app.py

import os, io, zipfile, tempfile, time, math
from collections import deque
from typing import List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

import cv2
import imageio.v3 as iio

# =============== helpers ===============
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1 + 1); ih = max(0, y2 - y1 + 1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    area_b = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter / float(area_a + area_b - inter)

def xyxy_to_cxcy(box):
    x1,y1,x2,y2 = box
    return (0.5*(x1+x2), 0.5*(y1+y2))

def draw_label(img, box, label, color=(0,255,255)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    rect_w = max(90, 9*len(label))
    top = max(0, y1-18)
    cv2.rectangle(img, (x1, top), (x1+rect_w, y1), color, -1)
    cv2.putText(img, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

class SimpleTracker:
    """Tiny IoU-greedy tracker with motion history."""
    def __init__(self, iou_thr=0.4, max_lost=20, history_len=90):
        self.next_id = 1
        self.tracks = {}
        self.iou_thr = iou_thr
        self.max_lost = max_lost
        self.history_len = history_len

    def update(self, dets: List[Tuple[List[float], str, float]]):
        boxes = [d[0] for d in dets]
        unmatched = set(range(len(dets)))

        # match
        for tid, t in list(self.tracks.items()):
            best_j = -1; best_iou = 0.0
            for j in list(unmatched):
                iou_ = iou(t['box'], boxes[j])
                if iou_ > best_iou:
                    best_iou, best_j = iou_, j
            if best_iou >= self.iou_thr:
                t['box'] = boxes[best_j]
                t['cls'] = dets[best_j][1]
                t['conf'] = dets[best_j][2]
                t['lost'] = 0
                # raw center history (we also keep stabilized history externally)
                t['history_raw'].append(xyxy_to_cxcy(boxes[best_j]))
                unmatched.discard(best_j)
            else:
                t['lost'] += 1

        # purge
        for tid in [tid for tid,t in self.tracks.items() if t['lost'] > self.max_lost]:
            del self.tracks[tid]

        # new tracks
        for j in unmatched:
            self.tracks[self.next_id] = {
                'box': boxes[j], 'cls': dets[j][1], 'conf': dets[j][2], 'lost': 0,
                'history_raw': deque([xyxy_to_cxcy(boxes[j])], maxlen=self.history_len),
                'history_stab': deque(maxlen=self.history_len), # stabilized centers
                'hist_h': deque(maxlen=self.history_len),       # bbox heights
                # behavior state
                'entered_roi_frame': None, 'loiter_alerted': False,
                'stationary_since': None, 'stationary_frames': 0,
                'last_person_near_frame': -10_000,
                # unusual
                'unusual_since': None, 'last_unusual_frame': -10_000,
                'unusual_fired': False, 'age_frames': 0
            }
            self.next_id += 1

        # age
        for t in self.tracks.values():
            if t['lost'] == 0: t['age_frames'] = t.get('age_frames', 0) + 1

        return self.tracks

class GlobalMotion:
    """Global translation via phase correlation (to stabilize camera shake/pan)."""
    def __init__(self, scale=0.5):
        self.prev = None
        self.scale = scale
        self.win = None
        self.cum = np.array([0.0, 0.0], dtype=np.float32)

    def update(self, frame_bgr):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        f32 = small.astype(np.float32)
        if self.prev is None:
            self.prev = f32
            self.win = cv2.createHanningWindow((f32.shape[1], f32.shape[0]), cv2.CV_32F)
            return self.cum.copy()  # zero shift initially
        # phase correlation (windowed)
        (dx, dy), _ = cv2.phaseCorrelate(self.prev * self.win, f32 * self.win)
        self.prev = f32
        # convert to full-res pixels (note: phaseCorrelate returns (dx, dy) in x,y order)
        self.cum += np.array([dx / self.scale, dy / self.scale], dtype=np.float32)
        return self.cum.copy()

def open_video_iter(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        def gen():
            while True:
                ok, f = cap.read()
                if not ok: break
                yield f
            cap.release()
        return gen(), fps, W, H, N
    props = iio.improps(path); fps = props.fps or 25.0; (W, H) = props.size; N = props.n_frames
    def gen():
        for frgb in iio.imiter(path):
            yield cv2.cvtColor(frgb, cv2.COLOR_RGB2BGR)
    return gen(), fps, W, H, N

def in_roi(box, roi_poly, W, H):
    if roi_poly is None or len(roi_poly) < 3: return True
    cx, cy = xyxy_to_cxcy(box)
    return cv2.pointPolygonTest(np.array(roi_poly, dtype=np.int32), (int(cx), int(cy)), False) >= 0

# =============== core pipeline ===============
def process_video(
    source_path: str,
    out_dir: str,
    weights: str = "yolov8n.pt",
    conf_thres: float = 0.45,
    # Loitering
    loiter_sec: float = 60.0,
    speed_px_per_sec_thr: float = 25.0,
    radius_px_thr: float = 20.0,
    # Unusual (conservative, running-focused)
    enable_unusual: bool = True,
    norm_speed_bls_thr: float = 1.5,  # body-lengths/sec threshold (running ~1.5–2.5)
    min_unusual_sec: float = 2.0,
    unusual_cooldown_sec: float = 4.0,
    one_unusual_per_track: bool = True,
    min_track_age_sec: float = 1.0,
    # Abandoned bag
    bag_stationary_sec: float = 25.0,
    owner_gap_sec: float = 12.0,
    # Filters
    min_box_area_ratio: float = 0.010,  # ignore tiny boxes (1% of frame)
    history_len_frames: int = 90,       # longer window → steadier stats
    # Misc
    progress_cb: Optional[Callable[[float], None]] = None,
    roi_poly: Optional[List[Tuple[int,int]]] = None
):
    os.makedirs(out_dir, exist_ok=True)
    snaps_dir = os.path.join(out_dir, "snaps"); os.makedirs(snaps_dir, exist_ok=True)

    model = YOLO(weights)
    frames, fps, W, H, N = open_video_iter(source_path)

    tracker = SimpleTracker(history_len=history_len_frames)
    gm = GlobalMotion(scale=0.5)

    loiter_frames = int(loiter_sec * fps)
    min_unusual_frames = int(min_unusual_sec * fps)
    unusual_cooldown_frames = int(unusual_cooldown_sec * fps)
    min_track_age_frames = int(min_track_age_sec * fps)
    bag_stationary_frames = int(bag_stationary_sec * fps)
    owner_gap_frames = int(owner_gap_sec * fps)
    near_dist_px = max(40, int(min(W, H) * 0.15))
    border_margin = int(0.03 * min(W, H))
    min_box_area_px = float(W * H) * float(min_box_area_ratio)

    person_names = {"person"}
    bag_names = {"backpack","handbag","suitcase"}

    events = []
    frame_idx = -1

    # simple EMA smoothing for stabilized centers
    def ema(prev, cur, alpha=0.6):
        return (alpha*prev[0] + (1-alpha)*cur[0], alpha*prev[1] + (1-alpha)*cur[1])

    for frame in frames:
        frame_idx += 1

        # 1) global camera stabilization (cumulative shift)
        cum_shift = gm.update(frame)  # (sx, sy) to subtract from centers

        # 2) detect persons & bags (filter small)
        res = model.predict(frame, imgsz=max(640, W), conf=conf_thres, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.cls.cpu().numpy(),
                               res.boxes.conf.cpu().numpy()):
                name = model.names[int(c)]
                if name in person_names or name in bag_names:
                    x1,y1,x2,y2 = b
                    if (x2-x1)*(y2-y1) < min_box_area_px:  # tiny/far objects → ignore
                        continue
                    dets.append((b.tolist(), name, float(s)))

        # 3) tracking
        tracks = tracker.update(dets)
        persons = [(tid, t) for tid,t in tracks.items() if t['cls'] in person_names and t['lost']==0]
        bags    = [(tid, t) for tid,t in tracks.items() if t['cls'] in bag_names and t['lost']==0]

        # 3a) append stabilized + smoothed centers and heights
        for _, t in tracks.items():
            cx, cy = xyxy_to_cxcy(t['box'])
            # subtract cumulative camera shift
            stab_c = (cx - cum_shift[0], cy - cum_shift[1])
            if len(t['history_stab']) == 0:
                t['history_stab'].append(stab_c)
            else:
                t['history_stab'].append(ema(t['history_stab'][-1], stab_c, alpha=0.6))
            h = float(t['box'][3] - t['box'][1])
            t['hist_h'].append(h)

        # 4) bag–person proximity for abandonment
        for btid, bt in bags:
            for ptid, pt in persons:
                bcx,bcy = xyxy_to_cxcy(bt['box']); pcx,pcy = xyxy_to_cxcy(pt['box'])
                if np.hypot(bcx-pcx, bcy-pcy) < near_dist_px:
                    bt['last_person_near_frame'] = frame_idx
                    break

        # 5) LOITERING (stationary dwell)
        for ptid, pt in persons:
            x1,y1,x2,y2 = map(int, pt['box'])
            if x1 < border_margin or y1 < border_margin or x2 > W-border_margin or y2 > H-border_margin:
                pt['entered_roi_frame'] = None; pt['loiter_alerted'] = False; pt['stationary_since'] = None
                continue

            inside = in_roi(pt['box'], roi_poly, W, H)
            hist = list(pt['history_stab'])[-45:]
            stationary = False
            if len(hist) >= 5:
                d = np.linalg.norm(np.diff(np.array(hist), axis=0), axis=1)  # px/frame (stabilized + smoothed)
                avg_speed = float(np.median(d)) * fps
                spread = np.linalg.norm(np.array(hist).max(0) - np.array(hist).min(0))
                stationary = (avg_speed < speed_px_per_sec_thr) and (spread < 2*radius_px_thr)

            if inside and stationary:
                if pt['stationary_since'] is None:
                    pt['stationary_since'] = frame_idx
            else:
                pt['stationary_since'] = None; pt['loiter_alerted'] = False

            if (pt['stationary_since'] is not None and not pt['loiter_alerted'] and
                frame_idx - pt['stationary_since'] >= loiter_frames):
                snap = frame.copy()
                draw_label(snap, pt['box'], f"LOITER id={ptid}")
                pth = os.path.join(snaps_dir, f"loiter_{frame_idx}_id{ptid}.jpg"); cv2.imwrite(pth, snap)
                events.append({"type":"loitering","video_time_sec":round(frame_idx/fps,2),"frame":frame_idx,
                               "track_id":ptid,"class":pt["cls"],"conf":round(pt["conf"],3),
                               "x1":int(pt["box"][0]),"y1":int(pt["box"][1]),"x2":int(pt["box"][2]),"y2":int(pt["box"][3]),
                               "snapshot":pth})
                pt['loiter_alerted'] = True

        # 6) UNUSUAL (running-style) — conservative
        if enable_unusual:
            for ptid, pt in persons:
                if one_unusual_per_track and pt.get('unusual_fired', False): 
                    continue
                if pt.get('age_frames', 0) < min_track_age_frames: 
                    continue
                if not in_roi(pt['box'], roi_poly, W, H):
                    continue

                # speeds on stabilized centers, normalized by body height (body-lengths/sec)
                centers = list(pt['history_stab'])
                heights = list(pt['hist_h'])
                if len(centers) < 10 or len(heights) < 10:
                    continue
                traj = np.array(centers[-45:])
                v = np.diff(traj, axis=0)                   # px/frame
                speed_pps = np.linalg.norm(v, axis=1) * fps # px/sec
                med_h = float(np.median(heights[-45:])) or 1.0
                bls = speed_pps / max(1.0, med_h)           # body-lengths/sec
                med_bls = float(np.median(bls))
                p90_bls = float(np.percentile(bls, 90))

                # require clearly high normalized speed (running), sustained
                is_fast = (p90_bls >= norm_speed_bls_thr) and (med_bls >= 0.8*norm_speed_bls_thr)

                if is_fast:
                    if pt.get('unusual_since') is None:
                        pt['unusual_since'] = frame_idx
                else:
                    pt['unusual_since'] = None

                long_enough = pt.get('unusual_since') is not None and \
                              (frame_idx - pt['unusual_since']) >= min_unusual_frames
                cooldown_ok = (frame_idx - pt.get('last_unusual_frame', -10_000)) >= unusual_cooldown_frames

                if long_enough and cooldown_ok:
                    snap = frame.copy()
                    draw_label(snap, pt['box'], f"UNUSUAL id={ptid}")
                    pth = os.path.join(snaps_dir, f"unusual_{frame_idx}_id{ptid}.jpg"); cv2.imwrite(pth, snap)
                    events.append({"type":"unusual_movement","video_time_sec":round(frame_idx/fps,2),"frame":frame_idx,
                                   "track_id":ptid,"class":pt["cls"],"conf":round(pt["conf"],3),
                                   "x1":int(pt["box"][0]),"y1":int(pt["box"][1]),"x2":int(pt["box"][2]),"y2":int(pt["box"][3]),
                                   "snapshot":pth})
                    pt['last_unusual_frame'] = frame_idx
                    if one_unusual_per_track: pt['unusual_fired'] = True

        # 7) Abandoned Bag
        for btid, bt in bags:
            hist = list(bt['history_stab'])
            if len(hist) >= 2:
                moved = np.linalg.norm(np.array(hist[-1]) - np.array(hist[-2]))
            else:
                moved = 0
            bt['stationary_frames'] = bt['stationary_frames'] + 1 if moved < 1.5 else 0
            owner_gap_ok = (frame_idx - bt['last_person_near_frame']) >= owner_gap_frames

            if bt['stationary_frames'] >= bag_stationary_frames and owner_gap_ok:
                snap = frame.copy()
                draw_label(snap, bt['box'], f"ABANDONED id={btid}")
                pth = os.path.join(snaps_dir, f"abandoned_{frame_idx}_id{btid}.jpg"); cv2.imwrite(pth, snap)
                events.append({"type":"abandoned_bag","video_time_sec":round(frame_idx/fps,2),"frame":frame_idx,
                               "track_id":btid,"class":bt["cls"],"conf":round(bt["conf"],3),
                               "x1":int(bt["box"][0]),"y1":int(bt["box"][1]),"x2":int(bt["box"][2]),"y2":int(bt["box"][3]),
                               "snapshot":pth})
                bt['stationary_frames'] = -999999  # cooldown

        if progress_cb and N:
            progress_cb(min(1.0, frame_idx / max(1, N)))

    df = pd.DataFrame(events).sort_values(["video_time_sec", "frame"]).reset_index(drop=True)
    csv_path = os.path.join(out_dir, "events.csv")
    df.to_csv(csv_path, index=False)
    return df, snaps_dir, csv_path

# =============== Streamlit UI ===============
st.set_page_config(page_title="AI Video Anomaly Detector", layout="wide")
st.title("🕵️ AI Video Anomaly Detector — robust unusual (running)")

with st.sidebar:
    st.header("⚙️ Settings")
    weights = st.text_input("YOLO weights", "yolov8n.pt")
    conf_thres = st.slider("Detection confidence", 0.1, 0.8, 0.45, 0.01)

    # Loitering
    loiter_sec = st.slider("Loitering seconds", 10, 180, 60, 1)
    speed_thr  = st.slider("Stationary speed < px/sec", 5, 80, 25, 1)
    radius_thr = st.slider("Stationary radius (px)", 5, 80, 20, 1)

    # Unusual (running) — conservative
    enable_unusual = st.checkbox("Enable unusual movement (running)", True)
    norm_speed_thr = st.slider("Normalized speed (body-lengths/sec)", 0.8, 3.0, 1.5, 0.1)
    min_unusual    = st.slider("Min unusual duration (sec)", 0.5, 5.0, 2.0, 0.1)
    cooldown       = st.slider("Cooldown between alerts (sec)", 0.0, 8.0, 4.0, 0.5)
    one_per_track  = st.checkbox("Only 1 unusual alert per person", True)
    min_track_age  = st.slider("Ignore tracks younger than (sec)", 0.0, 3.0, 1.0, 0.1)

    # Abandoned Bag
    bag_stat_sec  = st.slider("Bag stationary seconds", 5, 120, 25, 1)
    owner_gap_s   = st.slider("Owner-away seconds", 1, 30, 12, 1)

    # Filters
    min_box_area  = st.slider("Ignore small boxes (frame %)", 0.0, 5.0, 1.0, 0.1)

uploaded = st.file_uploader("Upload a video (mp4/avi/mov/mkv)", type=["mp4","avi","mov","mkv"])
run = st.button("▶️ Run Detection", disabled=(uploaded is None))

if run and uploaded:
    tmp_dir = tempfile.mkdtemp(prefix="anomaly_")
    src_path = os.path.join(tmp_dir, uploaded.name)
    with open(src_path, "wb") as f: f.write(uploaded.read())
    out_dir = os.path.join(tmp_dir, "outputs")
    st.info("Processing video…")

    prog = st.progress(0.0)
    def _cb(p): prog.progress(p)
    t0 = time.time()
    try:
        df, snaps_dir, csv_path = process_video(
            source_path=src_path, out_dir=out_dir, weights=weights, conf_thres=conf_thres,
            loiter_sec=loiter_sec, speed_px_per_sec_thr=speed_thr, radius_px_thr=radius_thr,
            enable_unusual=enable_unusual, norm_speed_bls_thr=norm_speed_thr,
            min_unusual_sec=min_unusual, unusual_cooldown_sec=cooldown,
            one_unusual_per_track=one_per_track, min_track_age_sec=min_track_age,
            bag_stationary_sec=bag_stat_sec, owner_gap_sec=owner_gap_s,
            min_box_area_ratio=min_box_area/100.0, progress_cb=_cb, roi_poly=None
        )
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()
    finally:
        prog.progress(1.0)

    st.success(f"Done in {time.time()-t0:.1f}s • {len(df)} events")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Loitering", int((df["type"]=="loitering").sum()))
    with c2: st.metric("Unusual Movement", int((df["type"]=="unusual_movement").sum()))
    with c3: st.metric("Abandoned Bag", int((df["type"]=="abandoned_bag").sum()))

    if len(df):
        st.subheader("Timeline")
        st.dataframe(df[["type","video_time_sec","frame","track_id","class","conf","snapshot"]],
                     use_container_width=True, height=320)

        st.subheader("Screenshots")
        cols = st.columns(4)
        for i, row in df.iterrows():
            snap = row["snapshot"]
            if isinstance(snap, str) and os.path.exists(snap):
                with cols[i % 4]:
                    st.image(snap, use_column_width=True,
                             caption=f"{row['type']} • t={row['video_time_sec']}s • id={row['track_id']}")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download events.csv", data=csv_bytes, file_name="events.csv", mime="text/csv")

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("events.csv", csv_bytes)
            for _, row in df.iterrows():
                snap = row["snapshot"]
                if isinstance(snap, str) and os.path.exists(snap):
                    zf.write(snap, arcname=os.path.join("snaps", os.path.basename(snap)))
        mem.seek(0)
        st.download_button("⬇️ Download snaps+csv.zip", data=mem,
                           file_name="anomalies_package.zip", mime="application/zip")
else:
    st.info("Upload a video and click **Run Detection**.")
