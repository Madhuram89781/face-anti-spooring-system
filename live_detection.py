# -*- coding: utf-8 -*-
# Live Face Anti-Spoofing Detection using OpenCV + MediaPipe + Silent-FAS Models

import os
import cv2
import numpy as np
import argparse
import warnings
import time

import mediapipe as mp
import torch
import torch.nn.functional as F

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

warnings.filterwarnings('ignore')

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class LiveFaceAntiSpoof:
    def __init__(self, model_dir, device_id=0):
        self.device = torch.device(
            "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        )
        self.model_dir = model_dir
        self.models = {}
        self._load_all_models()

        # MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def _load_all_models(self):
        """Pre-load all anti-spoof models for faster inference."""
        for model_name in os.listdir(self.model_dir):
            if not model_name.endswith('.pth'):
                continue
            model_path = os.path.join(self.model_dir, model_name)
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            kernel_size = get_kernel(h_input, w_input)
            model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device)
            keys = iter(state_dict)
            first_layer_name = next(keys)
            if first_layer_name.find('module.') >= 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    new_state_dict[key[7:]] = value
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
            model.eval()
            self.models[model_name] = {
                'model': model,
                'h_input': h_input,
                'w_input': w_input,
                'scale': scale
            }
        print(f"Loaded {len(self.models)} anti-spoof model(s)")

    def _get_bbox_from_mediapipe(self, image, detection):
        """Convert MediaPipe detection to [x, y, w, h] bbox."""
        h, w, _ = image.shape
        bbox = detection.location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = min(int(bbox.width * w), w - x)
        bh = min(int(bbox.height * h), h - y)
        return [x, y, bw, bh]

    def _crop_face(self, image, bbox, scale, out_w, out_h):
        """Crop face region with scale expansion."""
        src_h, src_w, _ = image.shape
        x, y, box_w, box_h = bbox

        if scale is None:
            return cv2.resize(image, (out_w, out_h))

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2 + x
        center_y = box_h / 2 + y

        left = int(max(0, center_x - new_width / 2))
        top = int(max(0, center_y - new_height / 2))
        right = int(min(src_w - 1, center_x + new_width / 2))
        bottom = int(min(src_h - 1, center_y + new_height / 2))

        img = image[top:bottom + 1, left:right + 1]
        return cv2.resize(img, (out_w, out_h))

    def predict(self, image, bbox):
        """Run anti-spoof prediction on a detected face."""
        prediction = np.zeros((1, 3))
        test_transform = trans.Compose([trans.ToTensor()])

        for info in self.models.values():
            img = self._crop_face(image, bbox, info['scale'], info['w_input'], info['h_input'])
            img_tensor = test_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                result = info['model'].forward(img_tensor)
                result = F.softmax(result, dim=1).cpu().numpy()
            prediction += result

        label = np.argmax(prediction)
        score = prediction[0][label] / len(self.models)
        is_real = (label == 1)
        return is_real, score

    def run(self, camera_id=0):
        """Main loop: capture webcam frames and perform live anti-spoofing."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Press 'q' to quit")

        fps_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = self._get_bbox_from_mediapipe(frame, detection)
                    x, y, w, h = bbox

                    if w < 20 or h < 20:
                        continue

                    is_real, score = self.predict(frame, bbox)

                    if is_real:
                        color = (0, 255, 0)
                        text = "REAL {:.2f}".format(score)
                    else:
                        color = (0, 0, 255)
                        text = "FAKE {:.2f}".format(score)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Calculate and display FPS
            current_time = time.time()
            if current_time - fps_time > 0:
                fps = 1.0 / (current_time - fps_time)
            fps_time = current_time

            cv2.putText(frame, "FPS: {:.1f}".format(fps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Live Face Anti-Spoofing", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.face_detection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Face Anti-Spoofing Detection")
    parser.add_argument("--device_id", type=int, default=0,
                        help="GPU device id (default: 0)")
    parser.add_argument("--model_dir", type=str,
                        default="./resources/anti_spoof_models",
                        help="Path to anti-spoof models directory")
    parser.add_argument("--camera_id", type=int, default=0,
                        help="Camera device id (default: 0)")
    args = parser.parse_args()

    detector = LiveFaceAntiSpoof(args.model_dir, args.device_id)
    detector.run(args.camera_id)

