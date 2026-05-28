# -*- coding: utf-8 -*-
"""셀프체크 결과 화면만 별도 캡처."""
import os
import sys
import asyncio
from playwright.async_api import async_playwright

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

URL = "http://localhost:8501"
OUT_DIR = "images"
VIEWPORT = {"width": 1600, "height": 1100}


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()

        await page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        await page.wait_for_load_state("networkidle", timeout=120000)
        await page.wait_for_timeout(12000)  # 초기 로드 + 모델 학습 대기

        # 사이드바: 교사 셀프체크 선택
        sidebar = page.locator('section[data-testid="stSidebar"]')
        await sidebar.locator('label').filter(has_text="교사 셀프체크").first.click()
        await page.wait_for_timeout(4000)

        # 제출 버튼 — 텍스트 기반으로 클릭
        await page.get_by_text("위험도 확인하기").click(timeout=15000)
        await page.wait_for_load_state("networkidle", timeout=60000)
        await page.wait_for_timeout(8000)  # Plotly 게이지·SHAP 차트 렌더 대기

        out = os.path.join(OUT_DIR, "teacher_selfcheck_result.png")
        await page.screenshot(path=out, full_page=True)
        print(f"saved: {out}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
