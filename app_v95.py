# -*- coding: utf-8 -*-
"""
IDDM V95 Streamlit Webアプリ（クラウド／ローカル両対応・相対パス完全版）
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path

# ページ設定
st.set_page_config(
    page_title="IDDM V95 血糖値状態判定（複数枚対応）",
    page_icon="🩺",
    layout="centered"
)

# スクリプト自身のディレクトリを基準とした相対パス（クラウド／ローカル共通）
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "results_v86_class" / "mobilenetv3_v86_classification_best.pth",
    BASE_DIR / "mobilenetv3_v86_classification_best.pth",
]

IMAGE_SIZE = 224
MODEL_NAME = "mobilenetv3_small_100"
DEVICE = torch.device("cpu")

# 顔検出器の安全なロード
face_cascade = None
try:
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if Path(face_cascade_path).exists():
        cascade = cv2.CascadeClassifier(face_cascade_path)
        if not cascade.empty():
            face_cascade = cascade
except Exception:
    face_cascade = None


def resolve_model_path():
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return MODEL_CANDIDATES[0]


MODEL_PATH = resolve_model_path()


def get_3class_label_text(class_id):
    mapping = {
        0: "正常範囲 (Normal: 70-179 mg/dL)",
        1: "高血糖 (Hyperglycemia: >=180 mg/dL)",
        2: "低血糖 (Hypoglycemia: <70 mg/dL)",
    }
    return mapping.get(class_id, "不明")


class Unified3ClassModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


@st.cache_resource
def load_model(model_path_str):
    target_path = Path(model_path_str)
    temp_backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    with torch.no_grad():
        feat_dim = temp_backbone(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape[1]

    model = Unified3ClassModel(feat_dim).to(DEVICE)
    loaded = False
    error_msg = None

    if target_path.exists():
        try:
            checkpoint = torch.load(target_path, map_location=DEVICE)
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)
            loaded = True
        except Exception as e:
            error_msg = str(e)
    else:
        searched = "\n".join(str(p.relative_to(BASE_DIR)) for p in MODEL_CANDIDATES)
        error_msg = f"モデルファイルが見つかりません。リポジトリ内に以下のいずれかの相対パスで配置してください:\n{searched}"

    model.eval()
    return model, loaded, error_msg


def robust_lighting_normalization(img_bgr):
    try:
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(y), cr, cb]), cv2.COLOR_YCrCb2BGR)
    except Exception:
        return img_bgr


def process_image_v57(img_bgr):
    h, w = img_bgr.shape[:2]
    if face_cascade is not None and not face_cascade.empty():
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(int(w * 0.25), int(h * 0.25))
            )
            if len(faces) > 0:
                (x, y, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
                if (fw >= w * 0.2) and (fh >= h * 0.2):
                    cx, cy = x + fw / 2.0, y + fh / 2.0
                    box_size = int(max(fw, fh) * 2.0)
                    x1, y1 = int(cx - box_size / 2.0), int(cy - box_size / 2.0)
                    cropped = img_bgr[max(0, y1):min(h, y1 + box_size), max(0, x1):min(w, x1 + box_size)]
                    if cropped.size > 0:
                        return robust_lighting_normalization(cropped), "Face_ROI_Zoomed"
        except Exception:
            pass

    min_dim = min(h, w)
    cx, cy = w / 2.0, h / 2.0
    box_size = int(min_dim * 0.95)
    x1, y1 = int(cx - box_size / 2.0), int(cy - box_size / 2.0)
    cropped = img_bgr[max(0, y1):min(h, y1 + box_size), max(0, x1):min(w, x1 + box_size)]
    if cropped.size == 0:
        cropped = img_bgr
    return robust_lighting_normalization(cropped), "Fallback_Zoomed"


# 状態保持メモリの初期化
if "last_stable_id" not in st.session_state:
    st.session_state.last_stable_id = None
if "recent_history" not in st.session_state:
    st.session_state.recent_history = []

model, model_loaded, model_error = load_model(str(MODEL_PATH))

st.title("🩺 IDDM V95 血糖値状態判定（複数枚対応）")
st.markdown("iPhoneのカメラロールやファイルから複数枚の画像を選択して、V93同様の安定化ロジックで一括判定します。")

col_btn1, col_btn2 = st.columns([1, 2])
with col_btn1:
    use_stabilization = st.checkbox("状態安定化を適用", value=True)
with col_btn2:
    if st.button("安定化履歴をリセット"):
        st.session_state.last_stable_id = None
        st.session_state.recent_history = []
        st.success("安定化履歴をリセットしました。")

if not model_loaded:
    st.error(f"モデル重みの読み込みに失敗しました:\n{model_error}")

uploaded_files = st.file_uploader(
    "顔写真（JPEG / PNG）を複数選択",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"選択された画像数: {len(uploaded_files)} 枚")

    local_last_stable_id = st.session_state.last_stable_id
    local_recent_history = list(st.session_state.recent_history)

    results = []
    with st.spinner("V95 ロバスト前処理＆複数枚推論を実行中..."):
        with torch.inference_mode():
            for idx, up_file in enumerate(uploaded_files, 1):
                file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    results.append({
                        "index": idx,
                        "name": up_file.name,
                        "error": "画像のデコードに失敗しました。"
                    })
                    continue

                processed_img, mode_str = process_image_v57(img_bgr)
                img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float() / 255.0
                tensor = tensor.unsqueeze(0).to(DEVICE)

                outputs = model(tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

                raw_id = int(np.argmax(probs))
                if use_stabilization:
                    stable_id = raw_id
                    if local_last_stable_id is not None and raw_id != local_last_stable_id:
                        sorted_probs = np.sort(probs)
                        conf_margin = float(sorted_probs[-1] - sorted_probs[-2])
                        if conf_margin < 0.25:
                            stable_id = local_last_stable_id

                    local_recent_history.append(stable_id)
                    if len(local_recent_history) > 5:
                        local_recent_history.pop(0)

                    final_id = int(np.bincount(local_recent_history).argmax())
                    local_last_stable_id = final_id
                else:
                    final_id = raw_id

                final_conf = float(probs[final_id])

                results.append({
                    "index": idx,
                    "name": up_file.name,
                    "error": None,
                    "img_rgb_display": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    "mode_str": mode_str,
                    "probs": probs,
                    "raw_id": raw_id,
                    "final_id": final_id,
                    "final_conf": final_conf,
                })

    if use_stabilization:
        st.session_state.last_stable_id = local_last_stable_id
        st.session_state.recent_history = local_recent_history

    st.markdown("---")
    st.subheader("📋 一括判定サマリー")

    summary_rows = []
    for r in results:
        if r["error"]:
            summary_rows.append({
                "No.": r["index"],
                "ファイル名": r["name"],
                "判定": "エラー",
                "確信度": "-",
                "正常": "-",
                "高血糖": "-",
                "低血糖": "-",
                "検出モード": "-",
            })
        else:
            p = r["probs"]
            summary_rows.append({
                "No.": r["index"],
                "ファイル名": r["name"],
                "判定": get_3class_label_text(r["final_id"]),
                "確信度": f"{r['final_conf'] * 100:.1f}%",
                "正常": f"{p[0] * 100:.1f}%",
                "高血糖": f"{p[1] * 100:.1f}%",
                "低血糖": f"{p[2] * 100:.1f}%",
                "検出モード": r["mode_str"],
            })

    st.table(summary_rows)

    st.markdown("---")
    st.subheader("🖼️ 各画像の詳細結果")

    for r in results:
        with st.expander(f"[{r['index']}/{len(results)}] {r['name']}", expanded=(len(results) <= 3)):
            if r["error"]:
                st.error(r["error"])
                continue

            st.image(r["img_rgb_display"], caption=r["name"], use_container_width=True)
            label_text = get_3class_label_text(r["final_id"])

            if r["final_id"] == 0:
                st.success(f"**{label_text}** (確信度: {r['final_conf'] * 100:.1f}%)")
            elif r["final_id"] == 1:
                st.warning(f"**{label_text}** (確信度: {r['final_conf'] * 100:.1f}%)")
            else:
                st.error(f"**{label_text}** (確信度: {r['final_conf'] * 100:.1f}%)")

            st.write(f"検出モード: `{r['mode_str']}` | 瞬時判定ID: `{r['raw_id']}` -> 最終判定ID: `{r['final_id']}`")

            p = r["probs"]
            c1, c2, c3 = st.columns(3)
            c1.metric("正常 (Normal)", f"{p[0] * 100:.1f}%")
            c2.metric("高血糖 (Hyper)", f"{p[1] * 100:.1f}%")
            c3.metric("低血糖 (Hypo)", f"{p[2] * 100:.1f}%")