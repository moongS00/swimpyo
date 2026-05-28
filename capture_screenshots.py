# -*- coding: utf-8 -*-
"""
Streamlit 앱 화면 자동 캡처 스크립트 (v2).
- Folium iframe 렌더 대기 강화
- 라디오 버튼 텍스트 매칭으로 사이드바 모드 전환
"""
import os
import sys
import asyncio
from playwright.async_api import async_playwright

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

URL = "http://localhost:8501"
OUT_DIR = "images"
VIEWPORT = {"width": 1600, "height": 1100}

os.makedirs(OUT_DIR, exist_ok=True)


async def wait_streamlit(page, ms=8000):
    await page.wait_for_load_state("networkidle", timeout=120000)
    await page.wait_for_timeout(ms)


async def wait_for_folium_map(page, timeout_ms=30000):
    """Folium 지도 iframe 내부의 leaflet 컨테이너가 그려질 때까지 대기."""
    try:
        # st_folium은 iframe[title="streamlit_folium.st_folium"] 또는 유사 패턴으로 렌더
        await page.wait_for_selector('iframe', timeout=timeout_ms)
        frames = page.frames
        for frame in frames:
            try:
                # leaflet 타일이 그려졌는지 확인
                el = await frame.wait_for_selector(".leaflet-container", timeout=5000)
                if el:
                    # 타일 로드 대기
                    await frame.wait_for_selector(".leaflet-tile-loaded", timeout=15000)
                    break
            except Exception:
                continue
        # 마커 클러스터 렌더 추가 대기
        await page.wait_for_timeout(4000)
    except Exception as e:
        print(f"[warn] folium wait: {e}")


async def click_tab(page, index):
    tabs = page.locator('button[role="tab"]')
    await tabs.nth(index).click()
    await page.wait_for_timeout(3000)


async def switch_to_teacher(page):
    """사이드바에서 '교사 셀프체크' 라디오 선택."""
    # Streamlit radio: 각 옵션은 label로 감싸지며 내부에 텍스트가 들어감
    sidebar = page.locator('section[data-testid="stSidebar"]')
    # 옵션 텍스트로 클릭
    teacher_option = sidebar.locator('label').filter(has_text="교사 셀프체크")
    await teacher_option.first.click()
    await page.wait_for_timeout(4000)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()

        # ============= 관리자 대시보드 =============
        await page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        # 초기 데이터 로드 + 모델 학습 대기 (~10초)
        await wait_streamlit(page, ms=12000)

        # 1) 전국 위험 지도 — Folium 충분히 로드되도록 대기
        await wait_for_folium_map(page, timeout_ms=30000)
        await page.screenshot(path=os.path.join(OUT_DIR, "admin_map.png"), full_page=True)
        print("saved: admin_map.png")

        # 2) 학교별 리포트 (탭 1)
        await click_tab(page, 1)
        await page.wait_for_timeout(4000)
        await page.screenshot(path=os.path.join(OUT_DIR, "admin_report.png"), full_page=True)
        print("saved: admin_report.png")

        # 3) What-if 시뮬레이터 (탭 2)
        await click_tab(page, 2)
        await page.wait_for_timeout(4000)
        await page.screenshot(path=os.path.join(OUT_DIR, "admin_whatif.png"), full_page=True)
        print("saved: admin_whatif.png")

        # ============= 교사 셀프체크 =============
        await switch_to_teacher(page)
        await wait_streamlit(page, ms=3000)
        await page.screenshot(path=os.path.join(OUT_DIR, "teacher_selfcheck_form.png"), full_page=True)
        print("saved: teacher_selfcheck_form.png")

        # 4) 제출 후 결과
        submit = page.locator('button[data-testid="stFormSubmitButton"]').first
        try:
            await submit.click(timeout=10000)
        except Exception:
            # 대체: form 안의 마지막 버튼
            await page.locator('form button').last.click()
        await page.wait_for_timeout(6000)
        await page.screenshot(path=os.path.join(OUT_DIR, "teacher_selfcheck_result.png"), full_page=True)
        print("saved: teacher_selfcheck_result.png")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
