# -*- coding: utf-8 -*-
"""
Pollinations.ai로 슬라이드 일러스트 5장 생성.
- 무료, 계정/API 키 불필요
- 공통 스타일 키워드로 화풍 일관성 확보
- seed 고정으로 재현성 보장
"""
import os
import sys
import urllib.parse
import urllib.request

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = "images"
os.makedirs(OUT_DIR, exist_ok=True)

# 공통 스타일 — 모든 일러스트에 추가하여 화풍 통일
STYLE = (
    "minimalist flat vector illustration, soft pastel palette of sage green and warm beige, "
    "clean lines, modern editorial style, professional, calm atmosphere, "
    "no text, no letters, no typography, wide landscape composition"
)

# 슬라이드별 프롬프트 (영문 — 한글 텍스트가 이미지에 들어가지 않도록)
ILLUSTRATIONS = [
    (
        "slide01_cover",
        "A delicate leaf shaped like a comma symbol with gentle radial light, "
        "a subtle semicircular risk gauge meter floating beside it, "
        "symbolizing rest and early warning",
    ),
    (
        "slide03_problem",
        "A weary teacher resting head on hand at a desk covered with stacks of paper and an empty coffee cup, "
        "soft window light, conveying burnout and exhaustion in an educational workplace",
    ),
    (
        "slide04_proposal",
        "An abstract grid of small school buildings arranged across a landscape, "
        "schools grouped by three subtle colors green yellow red representing risk levels, "
        "a soft glowing brain icon at the center connecting them with thin lines, "
        "conveying AI based predictive classification",
    ),
    (
        "slide13_teacher",
        "A young Korean teacher holding a smartphone showing a simple anonymous self check screen with a small gauge, "
        "a gentle leaf icon floating nearby, soft reassuring atmosphere, "
        "conveying privacy and easy access to help",
    ),
    (
        "slide15_closing",
        "A serene composition of a single leaf shaped like a comma with soft glowing radial lines, "
        "calm garden background out of focus, conveying rest and a pause, "
        "warm gentle finishing visual",
    ),
]

SEED = 42  # 재현성 고정


def fetch(prompt: str, filename: str):
    """Pollinations.ai 무료 익명 엔드포인트에서 이미지 다운로드."""
    full_prompt = f"{prompt}, {STYLE}"
    enc = urllib.parse.quote(full_prompt)
    # width/height는 무료 티어에서 자동 다운스케일링될 수 있음 — 그래도 비율 힌트 제공
    url = (
        f"https://image.pollinations.ai/prompt/{enc}"
        f"?width=1200&height=600&nologo=true&seed={SEED}&model=flux"
    )
    out = os.path.join(OUT_DIR, filename + ".png")
    print(f"[fetch] {filename} ... ", end="", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with open(out, "wb") as f:
        f.write(data)
    print(f"done ({len(data) // 1024} KB) → {out}")


def main():
    for name, prompt in ILLUSTRATIONS:
        try:
            fetch(prompt, name)
        except Exception as e:
            print(f"[error] {name}: {e}")


if __name__ == "__main__":
    main()
