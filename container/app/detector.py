import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(self, onnx_path="/app/models/best.onnx", imgsz=640, conf=0.25, class_names=("cat",)):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img, imgsz):
        """Resize image with padding to preserve aspect ratio."""
        orig_w, orig_h = img.size
        scale = min(imgsz / orig_w, imgsz / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Create padded image
        padded = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
        pad_x = (imgsz - new_w) // 2
        pad_y = (imgsz - new_h) // 2
        padded.paste(img_resized, (pad_x, pad_y))

        return padded, scale, (pad_x, pad_y)

    def predict(self, image_path: str) -> list:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # Letterbox preprocessing
        x, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        x = (np.array(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # Run inference
        out = self.session.run(None, {self.input_name: x})[0]  # (1, 300, 6)
        out = out[0]  # (300, 6)

        results = []
        for x1, y1, x2, y2, score, cls in out:
            if float(score) < self.conf:
                continue
            # Undo letterbox -> original image pixels
            x1 = (float(x1) - pad_x) / scale
            y1 = (float(y1) - pad_y) / scale
            x2 = (float(x2) - pad_x) / scale
            y2 = (float(y2) - pad_y) / scale
            # Clip to image bounds
            x1 = max(0.0, min(orig_w, x1))
            y1 = max(0.0, min(orig_h, y1))
            x2 = max(0.0, min(orig_w, x2))
            y2 = max(0.0, min(orig_h, y2))
            results.append({
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "confidence": float(score),
                "class": self.class_names[int(cls)],
            })
        return results
