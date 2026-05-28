# -*- coding: utf-8 -*-
"""모델·데이터 정보 페이지 캡처 + 사이드바 제외 크롭.
- streamlit-option-menu는 iframe 내부에 렌더되므로 frame 진입 필요.
"""
import os
import sys
import asyncio
from playwright.async_api import async_playwright
from PIL import Image

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

URL = "http://localhost:8501"
OUT_DIR = "images"
VIEWPORT = {"width": 1600, "height": 1400}


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()

        await page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        await page.wait_for_load_state("networkidle", timeout=120000)
        await page.wait_for_timeout(12000)

        # 사이드바의 option_menu iframe 진입
        iframe_handle = await page.locator('section[data-testid="stSidebar"] iframe').element_handle()
        frame = await iframe_handle.content_frame()
        assert frame is not None

        # frame 내부 a.nav-link만 한정 (li와 합쳐지면 인덱스가 꼬임)
        nav_links = frame.locator('a.nav-link')
        count = await nav_links.count()
        print(f"a.nav-link count: {count}")
        # 각 nav-link의 텍스트 확인 (디버그)
        for i in range(count):
            t = await nav_links.nth(i).inner_text()
            print(f"  [{i}] {t}")
        # 모델·데이터 정보 = 3번째 옵션 (index 2)
        await nav_links.nth(2).click()

        # 페이지 갱신 대기
        await page.wait_for_load_state("networkidle", timeout=60000)
        await page.wait_for_timeout(7000)

        full = os.path.join(OUT_DIR, "model_info_full.png")
        await page.screenshot(path=full, full_page=True)
        print(f"saved: {full}")
        await browser.close()

    # 사이드바 제외 크롭
    img = Image.open(full)
    cropped = img.crop((300, 0, img.width, img.height))
    out = os.path.join(OUT_DIR, "model_info_crop.png")
    cropped.save(out)
    print(f"cropped: {out} ({cropped.size})")


if __name__ == "__main__":
    asyncio.run(main())
