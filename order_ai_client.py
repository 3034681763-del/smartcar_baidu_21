import json
import os
import re
import time

import requests


QIANFAN_CHAT_URL = "https://qianfan.bj.baidubce.com/v2/chat/completions"


class OrderAIClient:
    """Parse OCR order text into a strict goods/building/name JSON result."""

    def __init__(
        self,
        api_key=None,
        model=None,
        api_url=None,
        timeout_s=8.0,
        max_retries=2,
    ):
        self.api_key = api_key or os.environ.get("QIANFAN_API_KEY")
        self.model = model or os.environ.get("QIANFAN_TEXT_MODEL", "ernie-4.5-8k-preview")
        self.api_url = api_url or os.environ.get("QIANFAN_CHAT_URL", QIANFAN_CHAT_URL)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)

    def parse_order(self, texts, valid_goods=None, valid_buildings=None, item_count=2):
        if not self.api_key:
            raise RuntimeError("QIANFAN_API_KEY is not set")

        payload = self._build_payload(
            texts,
            valid_goods=valid_goods,
            valid_buildings=valid_buildings,
            item_count=item_count,
        )
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
                return self._parse_result(
                    content,
                    valid_goods=valid_goods,
                    valid_buildings=valid_buildings,
                    item_count=item_count,
                )
            except Exception as exc:
                last_error = exc
                print(f"[OrderAI] parse failed {attempt}/{self.max_retries}: {exc}")
                time.sleep(0.5 * attempt)

        raise RuntimeError(f"Order AI parse failed: {last_error}")

    def _build_payload(self, texts, valid_goods=None, valid_buildings=None, item_count=2):
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        valid_goods = valid_goods or []
        valid_buildings = valid_buildings or []
        prompt = (
            "You are the order parser for an intelligent-car farm delivery task.\n"
            "The input comes from OCR and may contain wrong or missing characters.\n"
            "Parse the requested goods, building number, and receiver name.\n"
            "The order text will mention unit/building information, not abstract delivery addresses.\n"
            "The building field must be digits only, for example \"1\" or \"2\". Do not include Chinese words such as 单元.\n"
            "Return JSON only. Do not explain. Do not use markdown.\n"
            "The JSON schema must be exactly:\n"
            "{\"items\":[{\"goods\":\"...\",\"building\":\"...\",\"name\":\"...\"},{\"goods\":\"...\",\"building\":\"...\",\"name\":\"...\"}]}\n"
            f"Return exactly {int(item_count)} item(s).\n"
            f"Valid goods are: {', '.join(valid_goods) if valid_goods else 'unknown'}.\n"
            f"Valid building units are: {', '.join(valid_buildings) if valid_buildings else 'unknown'}.\n"
            "The name field should be the receiver/customer name from OCR, not the building unit.\n"
            "If a value cannot be determined, use null for that field.\n"
            f"OCR texts: {json.dumps([str(item) for item in texts], ensure_ascii=False)}"
        )
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 256,
            "temperature": 0.0,
        }

    @staticmethod
    def _parse_result(content, valid_goods=None, valid_buildings=None, item_count=2):
        text = str(content).strip()
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"no JSON object in AI response: {text}")
        data = json.loads(match.group(0))
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError(f"missing items list in AI response: {data}")

        valid_goods_set = {str(item).lower() for item in (valid_goods or [])}
        valid_building_set = {OrderAIClient._normalize_building(item) for item in (valid_buildings or [])}
        parsed = []
        for item in items[: int(item_count)]:
            if not isinstance(item, dict):
                continue
            goods = item.get("goods")
            building = item.get("building")
            name = item.get("name")
            goods = None if goods is None else str(goods).strip().lower()
            building = OrderAIClient._normalize_building(building)
            name = None if name is None else str(name).strip()
            if valid_goods_set and goods not in valid_goods_set:
                goods = None
            if valid_building_set and building not in valid_building_set:
                building = None
            parsed.append({"goods": goods, "building": building, "name": name})

        while len(parsed) < int(item_count):
            parsed.append({"goods": None, "building": None, "name": None})

        if any(item["goods"] is None or item["building"] is None or item["name"] is None for item in parsed):
            raise ValueError(f"incomplete parsed order: {parsed}")

        return {"items": parsed}

    @staticmethod
    def _normalize_building(value):
        if value is None:
            return None
        text = str(value).strip().upper()
        if not text:
            return None
        text = text.replace("單", "单")
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits or text
