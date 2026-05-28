# -*- coding: utf-8 -*-
"""사이드바 DOM 구조 디버깅."""
import sys
import asyncio
from playwright.async_api import async_playwright

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1600, "height": 1100})
        page = await context.new_page()

        await page.goto("http://localhost:8501", wait_until="domcontentloaded")
        await page.wait_for_load_state("networkidle", timeout=120000)
        await page.wait_for_timeout(10000)

        # 사이드바 iframe 확인
        sidebar = page.locator('section[data-testid="stSidebar"]')
        iframes = await sidebar.locator('iframe').count()
        print(f"iframes in sidebar: {iframes}")

        # iframe별 src + 크기 확인
        for i in range(iframes):
            ifr = sidebar.locator('iframe').nth(i)
            src = await ifr.get_attribute('src')
            print(f"  [{i}] src: {src[:120] if src else None}")

        # 사이드바 전체 텍스트
        text = await sidebar.inner_text()
        print(f"\n--- sidebar text ---\n{text}\n---")

        await browser.close()


asyncio.run(main())
