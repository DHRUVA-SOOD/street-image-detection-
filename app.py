
# STREAMLIT MULTIMODAL FUSION APP (FINAL VERSION)


import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="Multimodal Detection", layout="wide")
st.title("🚗 Multimodal Object Detection (RGB + Thermal + LiDAR)")


# LOAD MODELS (USE FULL PATH FOR NOW)


@st.cache_resource
def load_models():
    rgb_model = YOLO("https://huggingface.co/Daruvah/rgb_pt/resolve/main/rgb_final.pt")
    thermal_model = YOLO("https://huggingface.co/Daruvah/thermal_pt/resolve/main/thermal_final.pt")
    lidar_model = YOLO("https://huggingface.co/Daruvah/lidar_pt/resolve/main/lidar_final.pt")
    return rgb_model, thermal_model, lidar_model

rgb_model, thermal_model, lidar_model = load_models()


# CLASS NAMES


CLASS_NAMES = [
    "bicycle","bus","car","escooter",
    "motorcycle","person","tramway","truck"
]


# IOU


def iou(a,b):
    x1=max(a[0],b[0])
    y1=max(a[1],b[1])
    x2=min(a[2],b[2])
    y2=min(a[3],b[3])

    inter=max(0,x2-x1)*max(0,y2-y1)
    areaA=(a[2]-a[0])*(a[3]-a[1])
    areaB=(b[2]-b[0])*(b[3]-b[1])

    return inter/(areaA+areaB-inter+1e-6)


# GET BOXES


def get_boxes(res):
    boxes=[]
    for b in res.boxes:
        x1,y1,x2,y2=b.xyxy[0].cpu().numpy()
        conf=float(b.conf[0])
        cls=int(b.cls[0])
        boxes.append([x1,y1,x2,y2,conf,cls])
    return boxes


# WEIGHTED FUSION


def weighted_fusion(boxes,iou_thr=0.5):
    if len(boxes) == 0:
        return []

    fused=[]
    used=[False]*len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        group=[boxes[i]]
        used[i]=True

        for j in range(i+1,len(boxes)):
            if used[j]:
                continue
            if boxes[i][5]!=boxes[j][5]:
                continue
            if iou(boxes[i],boxes[j])>iou_thr:
                group.append(boxes[j])
                used[j]=True

        group=np.array(group)
        weights=group[:,4]

        x1=np.sum(group[:,0]*weights)/np.sum(weights)
        y1=np.sum(group[:,1]*weights)/np.sum(weights)
        x2=np.sum(group[:,2]*weights)/np.sum(weights)
        y2=np.sum(group[:,3]*weights)/np.sum(weights)

        conf=np.max(group[:,4])
        cls=int(group[0,5])

        fused.append([x1,y1,x2,y2,conf,cls])

    return fused


# DRAW BOXES


def draw(img,boxes):
    for b in boxes:
        x1,y1,x2,y2,conf,cls=b
        label=f"{CLASS_NAMES[cls]} {conf:.2f}"

        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(img,label,(int(x1),int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    return img


# LIDAR → DEPTH IMAGE


def lidar_to_depth(bin_path):
    pts = np.fromfile(bin_path,dtype=np.float32).reshape(-1,4)

    x,y,z = pts[:,0], pts[:,1], pts[:,2]
    depth = np.sqrt(x**2 + y**2 + z**2)

    W,H = 1024,1024
    img = np.zeros((H,W),dtype=np.float32)

    px = ((x+50)/100 * W).astype(int)
    py = ((y+50)/100 * H).astype(int)

    valid = (px>=0)&(px<W)&(py>=0)&(py<H)
    img[py[valid],px[valid]] = depth[valid]

    img = img/(img.max()+1e-6)
    img = (img*255).astype(np.uint8)
    img = np.stack([img,img,img],axis=-1)

    return img

# UI INPUTS


st.subheader("Upload Inputs (any combination)")

rgb_file = st.file_uploader("RGB Image", type=["jpg","png"])
thermal_file = st.file_uploader("Thermal Image", type=["jpg","png"])
lidar_file = st.file_uploader("LiDAR File (.bin)", type=["bin"])

# RUN DETECTION


if st.button("Run Detection"):

    if not rgb_file and not thermal_file and not lidar_file:
        st.warning("Please upload at least ONE input")
        st.stop()

    boxes = []
    base_image = None

    # -------- RGB --------
    if rgb_file:
        rgb = np.array(Image.open(rgb_file))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        base_image = rgb.copy()

        rgb_res = rgb_model(rgb)[0]
        boxes += get_boxes(rgb_res)

    # -------- THERMAL --------
    if thermal_file:
        thermal = np.array(Image.open(thermal_file))
        thermal = cv2.cvtColor(thermal, cv2.COLOR_RGB2BGR)

        if base_image is None:
            base_image = thermal.copy()

        thermal_res = thermal_model(thermal)[0]
        boxes += get_boxes(thermal_res)

    # -------- LIDAR --------
    if lidar_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(lidar_file.read())
            lidar = lidar_to_depth(tmp.name)

        if base_image is None:
            base_image = lidar.copy()

        lidar_res = lidar_model(lidar)[0]
        boxes += get_boxes(lidar_res)

    # -------- FUSION --------
    st.write(f"Detections before fusion: {len(boxes)}")

    fused = weighted_fusion(boxes)

    st.write(f"Detections after fusion: {len(fused)}")

    output = draw(base_image.copy(), fused)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    st.image(output, caption="Final Detection Output", use_container_width=True)
