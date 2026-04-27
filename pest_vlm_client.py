import base64
import json
import os
import re
import time

import cv2
import requests


QIANFAN_CHAT_URL = "https://qianfan.bj.baidubce.com/v2/chat/completions"


class PestVlmClient:
    """Baidu Qianfan multimodal client for one-card pest classification."""

    def __init__(
        self,
        api_key=None,
        model=None,
        api_url=None,
        timeout_s=8.0,
        max_retries=2,
        jpeg_quality=80,
    ):
        self.api_key = api_key or os.environ.get("QIANFAN_API_KEY")
        self.model = model or os.environ.get("QIANFAN_VLM_MODEL", "ernie-4.5-8k-preview")
        self.api_url = api_url or os.environ.get("QIANFAN_CHAT_URL", QIANFAN_CHAT_URL)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.jpeg_quality = int(jpeg_quality)

    def classify(self, image):
        if not self.api_key:
            raise RuntimeError("QIANFAN_API_KEY is not set")
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("empty animal crop")

        payload = self._build_payload(image)
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return self._parse_result(content)
            except Exception as exc:
                last_error = exc
                print(f"[PestVLM] classify failed {attempt}/{self.max_retries}: {exc}")
                time.sleep(0.5 * attempt)

        raise RuntimeError(f"Pest VLM classify failed: {last_error}")

    def _build_payload(self, image):
        image_b64 = self._encode_image(image)
        prompt = (
            "You are the image classification module for an intelligent-car farm pest-removal task.\n"
            "Judge whether the animal in this single card is harmful to crops.\n"
            "Harmful examples: locust, aphid, caterpillar, moth larva, crop pest, insect that damages crops.\n"
            "Beneficial examples: ladybug, bee, spider, frog, beneficial insect, animal that helps crops.\n"
            "Return JSON only. No explanation.\n"
            "If harmful, return exactly {\"result\":1}.\n"
            "If beneficial, return exactly {\"result\":0}."
        )
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "stream": False,
            "max_tokens": 64,
            "temperature": 0.0,
        }

    def _encode_image(self, image):
        ok, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode animal crop")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    @staticmethod
    def _parse_result(content):
        text = str(content).strip()
        match = re.search(r"\{.*?\}", text, flags=re.S)
        if not match:
            raise ValueError(f"no JSON object in VLM response: {text}")
        data = json.loads(match.group(0))
        result = int(data.get("result"))
        if result not in (0, 1):
            raise ValueError(f"invalid pest result: {result}")
        return result
