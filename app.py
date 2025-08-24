# AI Surveillance: Streamlit app for anomaly timeline + screenshots
# Usage:
#   pip install -r requirements.txt
#   streamlit run app.py

import os, io, zipfile, tempfile, time, math
from collections import deque
from typing import List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import cv2
import streamlit as st
from ultralytics import YOLO

# ----------------- Helpers -----------------
def iou(a, b):
    inter_x1 = max(a[0], b[0]); inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2]); inter_y2 = min(a[3], b[3])
    iw = max(0, inter_x2 - inter_x1 + 1); ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    area_b = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter / float(area_a + area_b - inter)

def xyxy_to_cxcy(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def draw_label(img, box, label, color=(0,255,255)):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    rect_w = max(90, 9*len(label))
    top = max(0, y1-18)
    cv2.rectangle(img, (x1, top), (x1+rect_w, y1), color, -1)
    cv2.putText(img, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

# Simple IoU-greedy tracker (lightweight)
class SimpleTracker:
    def __init__(self, iou_thr=0.4, max_lost=20, history_len=60):
        self.next_id = 1
        self.tracks = {}   # id -> state
        self.iou_thr = iou_thr
        self.max_lost = max_lost
        self.history_len = history_len

    def update(self, dets: List[Tuple[List[float], str, float]]):
        boxes = [d[0] for d in dets]
        unmatched = set(range(len(dets)))

        # try to match existing tracks
        for tid, t in list(self.tracks.items()):
            best_j = -1; best_iou = 0.0
            for j in list(unmatched):
                iou_ = iou(t['box'], boxes[j])
                if iou_ > best_iou:
                    best_iou, best_j = iou_, j
            if best_iou >= self.iou_thr:
                self.tracks[tid]['box'] = boxes[best_j]
                self.tracks[tid]['cls'] = dets[best_j][1]
                self.tracks[tid]['conf'] = dets[best_j][2]
                self.tracks[tid]['lost'] = 0
                self.tracks[tid]['history'].append(xyxy_to_cxcy(boxes[best_j]))
                unmatched.discard(best_j)
            else:
                self.tracks[tid]['lost'] += 1

        # purge lost tracks
        for tid in [tid for tid,t in self.tracks.items() if t['lost'] > self.max_lost]:
            del self.tracks[tid]

        # new tracks
        for j in unmatched:
            self.tracks[self.next_id] = {
                'box': boxes[j],
                'cls': dets[j][1],
                'conf': dets[j][2],
                'lost': 0,
                'history': deque([xyxy_to_cxcy(boxes[j])], maxlen=self.history_len),
                # behavior state
                'entered_roi_frame': None,
                'loiter_alerted': False,
                'stationary_since': None,
                'unusual_since': None,
                'last_unusual_frame': -10_000,
                'stationary_frames': 0,
                'last_person_near_frame': -10_000
            }
            self.next_id += 1

        return self.tracks

def in_roi(box, roi_poly, w, h):
    if roi_poly is None or len(roi_poly) < 3:
        return True
    cx, cy = xyxy_to_cxcy(box)
    return cv2.pointPolygonTest(np.array(roi_poly, dtype=np.int32), (int(cx), int(cy)), False) >= 0

# ----------------- Core processing -----------------
def process_video(
    source_path: str,
    out_dir: str,
    weights: str = "yolov8n.pt",
    conf_thres: float = 0.45,
    loiter_sec: float = 40.0,
    speed_px_per_sec_thr: float = 30.0,
    radius_px_thr: float = 25.0,
    history_win: int = 30,
    unusual_speed_thr: float = 160.0,
    heading_change_thr: float = 360.0,
    min_unusual_sec: float = 0.6,
    unusual_cooldown_sec: float = 2.0,
    bag_stationary_sec: float = 20.0,
    owner_gap_sec: float = 10.0,
    progress_cb: Optional[Callable[[float], None]] = None,
    roi_poly: Optional[List[Tuple[int,int]]] = None
):
    os.makedirs(out_dir, exist_ok=True)
    snaps_dir = os.path.join(out_dir, "snaps")
    os.makedirs(snaps_dir, exist_ok=True)

    model = YOLO(weights)
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    tracker = SimpleTracker()
    person_names = {"person"}
    bag_names = {"backpack", "handbag", "suitcase"}

    loiter_frames = int(loiter_sec * fps)
    min_unusual_frames = max(2, int(min_unusual_sec * fps))
    unusual_cooldown_frames = int(unusual_cooldown_sec * fps)
    bag_stationary_frames = int(bag_stationary_sec * fps)
    owner_gap_frames = int(owner_gap_sec * fps)
    near_dist_px = max(40, int(min(width, height) * 0.15))
    border_margin = int(0.03 * min(width, height))

    events = []
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Detect persons + bags
        res = model.predict(frame, imgsz=max(640, width), conf=conf_thres, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.cls.cpu().numpy(),
                               res.boxes.conf.cpu().numpy()):
                name = model.names[int(c)]
                if name in person_names or name in bag_names:
                    dets.append((b.tolist(), name, float(s)))

        tracks = tracker.update(dets)
        persons = [(tid, t) for tid,t in tracks.items() if t['cls'] in person_names and t['lost'] == 0]
        bags    = [(tid, t) for tid,t in tracks.items() if t['cls'] in bag_names and t['lost'] == 0]

        # bag-person proximity
        for btid, bt in bags:
            for ptid, pt in persons:
                if np.hypot(*(np.array(xyxy_to_cxcy(bt['box'])) - np.array(xyxy_to_cxcy(pt['box'])))) < near_dist_px:
                    bt['last_person_near_frame'] = frame_idx
                    break

        # -------- LOITERING (stationary dwell) --------
        for ptid, pt in persons:
            x1,y1,x2,y2 = map(int, pt['box'])
            # ignore if at edges (entrance/exit)
            if x1 < border_margin or y1 < border_margin or x2 > width-border_margin or y2 > height-border_margin:
                pt['entered_roi_frame'] = None
                pt['loiter_alerted'] = False
                pt['stationary_since'] = None
                continue

            inside = in_roi(pt['box'], roi_poly, width, height)
            hist = list(pt['history'])[-history_win:]
            stationary = False
            if len(hist) >= 5:
                d = np.linalg.norm(np.diff(np.array(hist), axis=0), axis=1)  # px/frame
                avg_speed = float(np.mean(d)) * fps                           # px/sec
                spread = np.linalg.norm(np.array(hist).max(0) - np.array(hist).min(0))
                stationary = (avg_speed < speed_px_per_sec_thr) and (spread < 2*radius_px_thr)

            if inside and stationary:
                if pt['stationary_since'] is None:
                    pt['stationary_since'] = frame_idx
            else:
                pt['stationary_since'] = None
                pt['loiter_alerted'] = False

            if (pt['stationary_since'] is not None and
                not pt['loiter_alerted'] and
                frame_idx - pt['stationary_since'] >= loiter_frames):
                snap = frame.copy()
                draw_label(snap, pt['box'], f"LOITER id={ptid}")
                snap_path = os.path.join(snaps_dir, f"loiter_{frame_idx}_id{ptid}.jpg")
                cv2.imwrite(snap_path, snap)
                events.append({
                    "type": "loitering",
                    "video_time_sec": round(frame_idx / fps, 2),
                    "frame": frame_idx,
                    "track_id": ptid,
                    "class": pt["cls"],
                    "conf": round(pt["conf"], 3),
                    "x1": int(pt["box"][0]), "y1": int(pt["box"][1]),
                    "x2": int(pt["box"][2]), "y2": int(pt["box"][3]),
                    "snapshot": snap_path
                })
                pt['loiter_alerted'] = True

        # -------- UNUSUAL MOVEMENT (fast / erratic) --------
        for ptid, pt in persons:
            inside = in_roi(pt['box'], roi_poly, width, height)
            hist = list(pt['history'])[-history_win:]
            is_unusual_now = False
            if inside and len(hist) >= 5:
                traj = np.array(hist)
                v = np.diff(traj, axis=0)                        # px/frame
                speed_pps = np.linalg.norm(v, axis=1) * fps      # px/sec
                # direction-change
                headings = np.array([math.atan2(dy, dx) for dx,dy in v])
                dh = np.diff(headings)
                dh = (dh + np.pi) % (2*np.pi) - np.pi
                dir_change_deg_per_sec = math.degrees(np.mean(np.abs(dh))) * fps if len(dh) else 0.0

                fast = float(np.mean(speed_pps)) > unusual_speed_thr
                erratic = dir_change_deg_per_sec > heading_change_thr and float(np.mean(speed_pps)) > 20
                is_unusual_now = fast or erratic

            if is_unusual_now:
                if pt['unusual_since'] is None:
                    pt['unusual_since'] = frame_idx
            else:
                pt['unusual_since'] = None

            if (pt['unusual_since'] is not None and
                frame_idx - pt['unusual_since'] >= min_unusual_frames and
                frame_idx - pt['last_unusual_frame'] >= unusual_cooldown_frames):
                snap = frame.copy()
                draw_label(snap, pt['box'], f"UNUSUAL id={ptid}")
                snap_path = os.path.join(snaps_dir, f"unusual_{frame_idx}_id{ptid}.jpg")
                cv2.imwrite(snap_path, snap)
                events.append({
                    "type": "unusual_movement",
                    "video_time_sec": round(frame_idx / fps, 2),
                    "frame": frame_idx,
                    "track_id": ptid,
                    "class": pt["cls"],
                    "conf": round(pt["conf"], 3),
                    "x1": int(pt["box"][0]), "y1": int(pt["box"][1]),
                    "x2": int(pt["box"][2]), "y2": int(pt["box"][3]),
                    "snapshot": snap_path
                })
                pt['last_unusual_frame'] = frame_idx

        # -------- ABANDONED BAG (stationary + no person near) --------
        for btid, bt in bags:
            if len(bt['history']) >= 2:
                (x1,y1) = bt['history'][-2]; (x2,y2) = bt['history'][-1]
                moved = np.hypot(x2-x1, y2-y1)
            else: moved = 0
            bt['stationary_frames'] = bt['stationary_frames'] + 1 if moved < 1.5 else 0
            owner_gap_ok = (frame_idx - bt['last_person_near_frame']) >= owner_gap_frames

            if bt['stationary_frames'] >= bag_stationary_frames and owner_gap_ok:
                snap = frame.copy()
                draw_label(snap, bt['box'], f"ABANDONED id={btid}")
                snap_path = os.path.join(snaps_dir, f"abandoned_{frame_idx}_id{btid}.jpg")
                cv2.imwrite(snap_path, snap)
                events.append({
                    "type": "abandoned_bag",
                    "video_time_sec": round(frame_idx / fps, 2),
                    "frame": frame_idx,
                    "track_id": btid,
                    "class": bt["cls"],
                    "conf": round(bt["conf"], 3),
                    "x1": int(bt["box"][0]), "y1": int(bt["box"][1]),
                    "x2": int(bt["box"][2]), "y2": int(bt["box"][3]),
                    "snapshot": snap_path
                })
                bt['stationary_frames'] = -999999   # cooldown

        if progress_cb and n_frames:
            progress_cb(min(1.0, frame_idx / max(1, n_frames)))

    cap.release()
    df = pd.DataFrame(events).sort_values(["video_time_sec", "frame"]).reset_index(drop=True)
    csv_path = os.path.join(out_dir, "events.csv")
    df.to_csv(csv_path, index=False)
    return df, snaps_dir, csv_path

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="AI Video Anomaly Detector", layout="wide")
st.title("🕵️ AI Video Anomaly Detector")
st.caption("Uploads a video → detects **Loitering**, **Unusual Movement**, and **Abandoned Bag**; "
           "outputs a timeline + screenshots with bounding boxes.")

with st.sidebar:
    st.header("⚙️ Settings")
    weights = st.text_input("YOLO weights", "yolov8n.pt")
    conf_thres = st.slider("Detection confidence", 0.1, 0.8, 0.45, 0.01)
    loiter_sec = st.slider("Loitering seconds", 10, 120, 40, 1)
    speed_thr  = st.slider("Stationary speed < px/sec", 5, 80, 30, 1)
    radius_thr = st.slider("Stationary radius (px)", 5, 80, 25, 1)
    unusual_speed = st.slider("Unusual speed > px/sec", 60, 400, 160, 5)
    heading_thr  = st.slider("Heading change > deg/sec", 90, 720, 360, 10)
    bag_stat_sec = st.slider("Bag stationary seconds", 5, 90, 20, 1)
    owner_gap_sec = st.slider("Owner-away seconds", 1, 30, 10, 1)

uploaded = st.file_uploader("Upload a video file (mp4/avi/mov/mkv)", type=["mp4","avi","mov","mkv"])
run = st.button("▶️ Run Detection", disabled=(uploaded is None))

if run and uploaded:
    # Save uploaded file to a temp path
    tmp_dir = tempfile.mkdtemp(prefix="anomaly_")
    src_path = os.path.join(tmp_dir, uploaded.name)
    with open(src_path, "wb") as f:
        f.write(uploaded.read())

    out_dir = os.path.join(tmp_dir, "outputs")
    st.info("Processing… this may take a bit depending on video length.")

    prog = st.progress(0.0)
    def _cb(p): prog.progress(p)

    t0 = time.time()
    try:
        df, snaps_dir, csv_path = process_video(
            source_path=src_path,
            out_dir=out_dir,
            weights=weights,
            conf_thres=conf_thres,
            loiter_sec=loiter_sec,
            speed_px_per_sec_thr=speed_thr,
            radius_px_thr=radius_thr,
            unusual_speed_thr=unusual_speed,
            heading_change_thr=heading_thr,
            bag_stationary_sec=bag_stat_sec,
            owner_gap_sec=owner_gap_sec,
            progress_cb=_cb,
            roi_poly=None  # set polygon list if you want ROI limitation
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    finally:
        prog.progress(1.0)

    st.success(f"Done in {time.time()-t0:.1f}s • {len(df)} events")
    # Summary
    colA, colB, colC = st.columns(3)
    with colA: st.metric("Loitering", int((df["type"]=="loitering").sum()))
    with colB: st.metric("Unusual Movement", int((df["type"]=="unusual_movement").sum()))
    with colC: st.metric("Abandoned Bag", int((df["type"]=="abandoned_bag").sum()))

    # Timeline table
    if len(df):
        st.subheader("Timeline")
        st.dataframe(df[["type","video_time_sec","frame","track_id","class","conf","snapshot"]],
                     use_container_width=True, height=320)

        # Gallery
        st.subheader("Screenshots")
        n_cols = 4
        cols = st.columns(n_cols)
        for i, row in df.iterrows():
            snap = row["snapshot"]
            if isinstance(snap, str) and os.path.exists(snap):
                with cols[i % n_cols]:
                    st.image(snap, use_column_width=True,
                             caption=f"{row['type']} • t={row['video_time_sec']}s • id={row['track_id']}")

        # Download buttons
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download events.csv", data=csv_bytes, file_name="events.csv", mime="text/csv")

        # Zip screenshots + CSV
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("events.csv", csv_bytes)
            for _, row in df.iterrows():
                snap = row["snapshot"]
                if isinstance(snap, str) and os.path.exists(snap):
                    zf.write(snap, arcname=os.path.join("snaps", os.path.basename(snap)))
        mem.seek(0)
        st.download_button("⬇️ Download snaps+csv.zip", data=mem, file_name="anomalies_package.zip", mime="application/zip")

    else:
        st.warning("No anomalies found with the current thresholds. Try relaxing thresholds or using another video.")
else:
    st.info("Upload a video and click **Run Detection** to begin.")
