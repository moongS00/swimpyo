# -*- coding: utf-8 -*-
"""
쉼표(Swimpyo) — AI 기반 교사 업무·감정 부담 조기경보 시스템 (대회 출품 프로토타입)

[대회]
    제8회 교육 공공데이터 AI 활용대회

[활용 공공데이터 (data/ 하위 7종, 경기도교육청, 2022~2025)]
    1) 학교기본정보(초/중/고)_경기도교육청.csv
       — 시도교육청·교육지원청·지역·정보공시 학교코드·학교명·학교급코드·위도·경도
    2) 직위별 교원 현황(초/중/고)_경기도교육청.csv
       — 기간제교사(계), 휴직교원수, 총계(계) 등
    3) 자격종별 교원 현황(초/중/고)_경기도교육청.csv
       — 정교사(1정·2정)(계), 계 등
    4) 학년별·학급별 학생수(초/중/고)_경기도교육청.csv
       — 학급당 학생수(계), 수업교원 1인당 학생수, 학생수(계) 등
    5) 수업일수 및 수업시수 현황(초/중/고)_경기도교육청.csv
       — 주당평균수업시수(교사 1인당) 등
    6) 학생·학부모 상담계획 및 실시 현황(초/중/고)_경기도교육청.csv
       — 상담실적(내부/외부), WEE클래스 설치여부 등
    7) 동아리 활동 현황(초/중/고)_경기도교육청.csv
       — 동아리수, 지도교사수 (정규/비정규 두 셋의 합)

[모델 선택 근거]
    LightGBM 회귀 모델을 채택했다. 이유는 다음과 같다.
    (1) 메모리 효율: Streamlit Community Cloud Free(1GB RAM)에서 안정 동작.
        XGBoost 대비 트리당 메모리 사용량이 작고 학습이 빠르다.
    (2) SHAP TreeExplainer와 직접 호환: 의사결정 근거(설명가능성)를
        대회 핵심 차별점으로 가져가기에 적합하다.
    (3) 카테고리·수치 혼합 피처를 자연스럽게 처리하며 결측치도 내부 처리한다.
    (4) ACAD 검증 보고서 ADR-007에서 확정된 알고리즘이다.

[정답 데이터 부재 → 합성 TBRI 점수 사용]
    공공데이터에는 "교사 번아웃 실측" 라벨이 없다. 따라서 도메인 휴리스틱 기반
    가중합으로 합성 TBRI 점수를 만든 뒤, 이를 회귀 타겟으로 LightGBM을 학습시켰다.
    이는 (a) 파이프라인 전체(전처리→학습→추론→SHAP)를 실제 데이터로 구동하는
    프로토타입 검증이 목적이며, (b) 실제 서비스화 시 시도교육청 협력 데이터로
    라벨을 교체할 수 있도록 인터페이스를 유지했다.

[익명화 정책 (필수 요구사항 6)]
    데이터 로드 직후 학교명·지역·교육지원청을 모두 익명 라벨로 치환한다.
    원본은 어떤 화면에서도 노출하지 않는다.
        학교명 → "초_0001", "중_0001", "고_0001" 형식 (학교급별 일련번호)
        지역  → "지역_A", "지역_B", ... (가나다 정렬 후 알파벳)
        교육지원청 → "교육청_A", "교육청_B", ...
    위도·경도는 지도 표시에 필요하므로 유지하되, 지도의 모든 라벨·툴팁은
    익명 라벨만 사용한다.

[캐싱 전략]
    @st.cache_data       : 정적 데이터(CSV 로드·전처리 결과 DataFrame)
    @st.cache_resource   : 모델·SHAP Explainer(불변, 직렬화 비용 절감)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import folium
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE = "쉼표(Swimpyo)"
APP_SUBTITLE = "AI 기반 교사 업무·감정 부담 조기경보 시스템"
TARGET_YEAR = "2024"  # 분석 기준 연도 (가장 완전한 최근 연도)
DATA_DIR = Path(__file__).parent / "data"

# 카테고리 키 → 폴더명. 7개 모두 정보공시 학교코드로 조인 가능.
CATEGORIES = {
    "base": "학교기본정보",
    "position": "직위별 교원 현황",
    "license": "자격종별 교원 현황",
    "classroom": "학년별 학급별 학생수",
    "teach_hours": "수업일수 및 수업시수 현황",
    "counseling": "학생 학부모 상담계획 및 실시현황",
    "club": "동아리 활동현황",
}

# 학교급 코드 매핑 (학교알리미 공시 표준)
LEVEL_MAP = {"02": "초", "03": "중", "04": "고"}

# 학습/추론에 사용하는 피처 컬럼 (전처리 후 마스터 테이블 내부 명)
FEATURE_COLS = [
    "학급당 학생수(계)",
    "수업교원 1인당 학생수",
    "주당평균수업시수(교사 1인당)",
    "기간제_비율",
    "휴직_비율",
    "정교사_비율",
    "상담_총건수",
    "WEE_미설치",
    "동아리수_합",
    "학생수(계)",
]

# 화면 표시용 한글 라벨
FEATURE_LABELS_KOR = {
    "학급당 학생수(계)": "학급당 학생수",
    "수업교원 1인당 학생수": "교사 1인당 학생수",
    "주당평균수업시수(교사 1인당)": "주당 평균 수업시수",
    "기간제_비율": "기간제 교사 비율",
    "휴직_비율": "교원 휴직률",
    "정교사_비율": "정교사 비율",
    "상담_총건수": "연간 상담 건수",
    "WEE_미설치": "WEE클래스 미설치(0=설치,1=미설치)",
    "동아리수_합": "동아리 운영 수(정규+비정규)",
    "학생수(계)": "전체 학생 수",
}

# 합성 TBRI 점수 가중치 (도메인 휴리스틱)
# 양수 = 값이 클수록 위험 상승, 음수 = 값이 클수록 위험 하락
PROXY_WEIGHTS = {
    "학급당 학생수(계)": 0.18,
    "수업교원 1인당 학생수": 0.18,
    "주당평균수업시수(교사 1인당)": 0.14,
    "기간제_비율": 0.14,
    "휴직_비율": 0.14,
    "정교사_비율": -0.10,
    "상담_총건수": 0.05,
    "WEE_미설치": 0.04,
    "동아리수_합": -0.05,
    "학생수(계)": 0.02,
}

# SHAP 설명 시 사용할 자연어 변환 사전 (피처 → 비전문가 설명)
RISK_NARRATIVE = {
    "학급당 학생수(계)": "한 학급의 학생이 많을수록 담임 부담이 누적됩니다.",
    "수업교원 1인당 학생수": "교사 1인당 담당 학생 수가 많으면 행정·상담 부담이 커집니다.",
    "주당평균수업시수(교사 1인당)": "주당 수업시수가 많으면 수업 준비·평가 부담이 커집니다.",
    "기간제_비율": "기간제 교사 비율이 높으면 인력 안정성이 낮아 부담이 정규직에 쏠립니다.",
    "휴직_비율": "교원 휴직률은 학교 환경의 누적 스트레스를 나타내는 신호입니다.",
    "정교사_비율": "정교사 비율은 학교의 인력 안정성을 의미합니다 (높을수록 안정).",
    "상담_총건수": "상담 건수가 많은 학교는 학생·학부모 정서 부담이 큰 환경입니다.",
    "WEE_미설치": "WEE클래스 미설치는 학교 내 정서 지원 자원의 부재를 의미합니다.",
    "동아리수_합": "동아리 활동이 활발한 학교는 학생 참여·관계망이 풍부합니다 (보호 요인).",
    "학생수(계)": "학교 규모가 클수록 행정 복잡도가 증가합니다.",
}


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로딩 (@st.cache_data 로 캐싱)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_category(category_key: str, year: str = TARGET_YEAR) -> pd.DataFrame:
    """
    data/<카테고리 폴더>/ 의 CSV 파일을 모두 읽어 단일 DataFrame으로 반환한다.
    - 학교기본정보는 연도 무관 (단일 스냅샷).
    - 그 외는 `{year}년도_*.csv` 패턴으로 초/중/고 3개 파일을 모두 로드해 concat.
    - 모든 컬럼을 일단 문자열(dtype=str)로 읽어 mixed-type 경고를 피하고,
      수치 변환은 build_master_table()에서 명시적으로 수행한다.
    - 인코딩은 UTF-8 BOM 우선, 실패 시 cp949로 폴백.
    """
    folder = DATA_DIR / CATEGORIES[category_key]
    if not folder.exists():
        return pd.DataFrame()

    if category_key == "base":
        files = sorted(folder.glob("*.csv"))
    else:
        files = sorted(folder.glob(f"{year}년도_*.csv"))

    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", dtype=str, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="cp949", dtype=str, low_memory=False)
        df["__source_file"] = f.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _to_num(series: pd.Series) -> pd.Series:
    """문자열 수치를 안전하게 float으로 변환."""
    return pd.to_numeric(series, errors="coerce")


@st.cache_data(show_spinner=False)
def build_master_table(year: str = TARGET_YEAR) -> pd.DataFrame:
    """
    7개 카테고리를 '정보공시 학교코드'로 좌측 조인하여 학교 단위 마스터 테이블 생성.

    전처리 단계:
    1) 폐교·휴교 학교 제거 (학교기본정보의 폐교여부/휴교여부 필터)
    2) 카테고리별 핵심 컬럼만 추출 후 수치 변환
    3) 파생 피처 계산:
       - 기간제_비율 = 기간제교사(계) / 총계(계)
       - 휴직_비율  = 휴직교원수 / 총계(계)
       - 정교사_비율 = (정교사1정 + 정교사2정) / 자격계
       - 상담_총건수 = 상담실적(내부) + 상담실적(외부)
       - WEE_미설치  = (교내 WEE클래스 설치여부 != 'Y')
       - 동아리수_합 = 정규동아리수 + 비정규동아리수
    4) 결측치는 호출 측에서 중앙값 대체 (모델 학습 직전).
    """
    base = load_category("base")
    if base.empty:
        st.error(
            f"data/{CATEGORIES['base']} 폴더에서 CSV를 찾지 못했습니다. "
            "data/ 디렉토리 구조를 확인하세요."
        )
        st.stop()

    pos = load_category("position", year)
    lic = load_category("license", year)
    cls = load_category("classroom", year)
    th = load_category("teach_hours", year)
    co = load_category("counseling", year)
    club = load_category("club", year)

    # ── 1) 학교기본정보: 폐교·휴교 필터, 좌표 수치화
    base = base.copy()
    base = base[base.get("폐교여부", "N").fillna("N") != "Y"]
    if "휴교여부" in base.columns:
        base = base[base["휴교여부"].fillna("N") != "Y"]

    base["위도"] = _to_num(base["위도"])
    base["경도"] = _to_num(base["경도"])

    key = "정보공시 학교코드"
    keep_base = [
        "시도교육청", "교육지원청", "지역", key, "학교명",
        "학교급코드", "설립구분", "위도", "경도",
    ]
    df = base[keep_base].drop_duplicates(subset=[key]).copy()

    # ── 2) 직위별 교원 현황: 기간제/휴직/총계
    if not pos.empty:
        for c in ("기간제교사(계)", "휴직교원수", "총계(계)"):
            if c in pos.columns:
                pos[c] = _to_num(pos[c])
        pos_keep = pos[[key, "기간제교사(계)", "휴직교원수", "총계(계)"]].copy()
        pos_keep = pos_keep.drop_duplicates(subset=[key])
        df = df.merge(pos_keep, on=key, how="left")

        # 안전한 분모 처리 (총계=0인 학교는 결측 처리)
        denom = df["총계(계)"].replace(0, np.nan)
        df["기간제_비율"] = (df["기간제교사(계)"] / denom).clip(0, 1)
        df["휴직_비율"] = (df["휴직교원수"] / denom).clip(0, 1)
    else:
        df["기간제_비율"] = np.nan
        df["휴직_비율"] = np.nan

    # ── 3) 자격종별 교원 현황: 정교사 비율
    if not lic.empty:
        for c in ("정교사(1정)(계)", "정교사(2정)(계)", "계"):
            if c in lic.columns:
                lic[c] = _to_num(lic[c])
        denom_l = lic["계"].replace(0, np.nan) if "계" in lic.columns else np.nan
        lic["정교사_비율"] = (
            (lic.get("정교사(1정)(계)", 0).fillna(0) + lic.get("정교사(2정)(계)", 0).fillna(0))
            / denom_l
        ).clip(0, 1)
        df = df.merge(lic[[key, "정교사_비율"]].drop_duplicates(subset=[key]),
                      on=key, how="left")
    else:
        df["정교사_비율"] = np.nan

    # ── 4) 학년별 학급별 학생수: 학급당·교사 1인당·학생수
    if not cls.empty:
        cls_cols = ["학급수(계)", "학생수(계)", "학급당 학생수(계)",
                    "교사수", "수업교원 1인당 학생수"]
        for c in cls_cols:
            if c in cls.columns:
                cls[c] = _to_num(cls[c])
        df = df.merge(
            cls[[key] + [c for c in cls_cols if c in cls.columns]].drop_duplicates(subset=[key]),
            on=key, how="left",
        )
    else:
        for c in ("학급수(계)", "학생수(계)", "학급당 학생수(계)",
                  "교사수", "수업교원 1인당 학생수"):
            df[c] = np.nan

    # ── 5) 수업시수
    if not th.empty and "주당평균수업시수(교사 1인당)" in th.columns:
        th["주당평균수업시수(교사 1인당)"] = _to_num(th["주당평균수업시수(교사 1인당)"])
        df = df.merge(
            th[[key, "주당평균수업시수(교사 1인당)"]].drop_duplicates(subset=[key]),
            on=key, how="left",
        )
    else:
        df["주당평균수업시수(교사 1인당)"] = np.nan

    # ── 6) 상담·WEE클래스
    if not co.empty:
        in_col = "상담실적(내부상담전문가)"
        ex_col = "상담실적(외부상담전문가)"
        we_col = "교내 WEE클래스 설치여부"
        if in_col in co.columns:
            co[in_col] = _to_num(co[in_col]).fillna(0)
        if ex_col in co.columns:
            co[ex_col] = _to_num(co[ex_col]).fillna(0)
        co["상담_총건수"] = co.get(in_col, 0) + co.get(ex_col, 0)
        co["WEE_미설치"] = (co.get(we_col, "N").fillna("N") != "Y").astype(int)
        df = df.merge(
            co[[key, "상담_총건수", "WEE_미설치"]].drop_duplicates(subset=[key]),
            on=key, how="left",
        )
    else:
        df["상담_총건수"] = np.nan
        df["WEE_미설치"] = np.nan

    # ── 7) 동아리: pandas는 중복 컬럼명을 자동으로 '동아리수' / '동아리수.1' 로 변경
    if not club.empty:
        for c in ("동아리수", "동아리수.1", "지도교사수", "지도교사수.1"):
            if c in club.columns:
                club[c] = _to_num(club[c]).fillna(0)
        club["동아리수_합"] = club.get("동아리수", 0) + club.get("동아리수.1", 0)
        df = df.merge(
            club[[key, "동아리수_합"]].drop_duplicates(subset=[key]),
            on=key, how="left",
        )
    else:
        df["동아리수_합"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 익명화 (필수 요구사항 6)
# 데이터 로드 직후 학교명·지역·교육지원청을 익명 라벨로 치환.
# 원본 학교명은 모든 화면 출력에서 제거된다.
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_label(idx: int) -> str:
    """0→A, 25→Z, 26→AA, 27→AB ... 알파벳 라벨 생성."""
    label = ""
    n = idx
    while True:
        label = chr(ord("A") + (n % 26)) + label
        n = n // 26 - 1
        if n < 0:
            break
    return label


def anonymize(df: pd.DataFrame) -> pd.DataFrame:
    """
    학교명·지역·교육지원청을 익명 라벨로 치환한다.
    원본 컬럼은 모두 제거하며, 이후 어떤 화면도 익명 라벨만 본다.
    """
    df = df.copy()

    # 학교명 익명: 학교급별로 정렬하여 순번 부여 → "초_0001", "중_0001", "고_0001"
    df["학교급_라벨"] = df["학교급코드"].astype(str).str.zfill(2).map(LEVEL_MAP).fillna("기타")
    df = df.sort_values(["학교급_라벨", "정보공시 학교코드"]).reset_index(drop=True)
    df["순번"] = df.groupby("학교급_라벨").cumcount() + 1
    df["익명_학교명"] = df["학교급_라벨"] + "_" + df["순번"].apply(lambda x: f"{x:04d}")
    df = df.drop(columns=["순번"])

    # 지역(시군구) 익명: 가나다 정렬 후 A, B, ...
    regions = sorted(df["지역"].dropna().unique())
    region_map = {r: f"지역_{_alpha_label(i)}" for i, r in enumerate(regions)}
    df["익명_지역"] = df["지역"].map(region_map)

    # 교육지원청 익명
    offices = sorted(df["교육지원청"].dropna().unique())
    office_map = {o: f"교육청_{_alpha_label(i)}" for i, o in enumerate(offices)}
    df["익명_교육지원청"] = df["교육지원청"].map(office_map)

    # 원본 식별 컬럼 제거 (학교명, 지역명, 교육지원청 원문은 어디서도 노출되지 않음)
    df = df.drop(columns=["학교명", "지역", "교육지원청"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 모델 학습 & SHAP (@st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

def compute_proxy_tbri(feat_df: pd.DataFrame) -> pd.Series:
    """
    합성 TBRI 점수 (0~100, 백분위 정규화).
    공공데이터에 교사 번아웃 실측이 없으므로 도메인 휴리스틱 가중합을 사용한다.
    PROXY_WEIGHTS의 부호가 위험 방향을 나타낸다.
    실제 서비스화 시 시도교육청 협력 데이터로 라벨을 교체할 수 있다.
    """
    z = feat_df.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    score_z = sum(z[c] * w for c, w in PROXY_WEIGHTS.items() if c in z.columns)
    # 백분위 순위 (0~100) — 절대값에 의존하지 않아 해석이 단순
    return (score_z.rank(pct=True) * 100).clip(0, 100)


def risk_grade(score: float) -> str:
    if score < 34:
        return "녹색(저위험)"
    if score < 67:
        return "황색(주의)"
    return "적색(고위험)"


def risk_color(score: float) -> str:
    if score < 34:
        return "#2ca02c"
    if score < 67:
        return "#ff9800"
    return "#d62728"


@st.cache_data(show_spinner="공공데이터 적재·전처리·익명화 중…")
def get_master_with_scores(year: str = TARGET_YEAR) -> Tuple[pd.DataFrame, dict]:
    """
    마스터 테이블 + 결측치 보정 + 합성 TBRI 점수 + 익명화까지 한 번에 수행.
    반환:
        df : 피처가 채워지고 익명 라벨이 부여된 학교별 DataFrame
        medians : 피처별 중앙값 (셀프체크 imputation 기본값으로도 사용)
    """
    df = build_master_table(year)

    # ── 결측치 처리: 학습 안정성을 위해 중앙값으로 imputation
    # (피처 분포가 long-tail이므로 평균보다 중앙값이 안전)
    feat_df = df[FEATURE_COLS].copy()
    medians = feat_df.median(numeric_only=True).to_dict()
    feat_df = feat_df.fillna(value=medians)

    # ── 스케일링: 트리 기반 모델(LightGBM)은 스케일 불변이라 표준화 불필요.
    #    단, SHAP 시각화의 해석을 위해 원본 단위를 그대로 사용한다.
    #    합성 라벨 산출에만 z-score를 임시로 사용한다(compute_proxy_tbri 내부).
    for c in FEATURE_COLS:
        df[c] = feat_df[c]  # 채워진 값으로 갱신

    df["tbri_score"] = compute_proxy_tbri(feat_df)
    df["risk_grade"] = df["tbri_score"].apply(risk_grade)

    df = anonymize(df)
    return df, medians


@st.cache_resource(show_spinner="LightGBM 모델 학습 + SHAP Explainer 생성 중…")
def get_model_and_explainer():
    """
    LightGBM 회귀 모델 학습 + SHAP TreeExplainer 생성.
    cache_resource로 앱 부팅 시 1회만 학습하며 모든 세션이 공유한다.
    """
    df, medians = get_master_with_scores(TARGET_YEAR)
    X = df[FEATURE_COLS].values
    y = df["tbri_score"].values

    # 하이퍼파라미터: ACAD 검증 보고서 ADR-002/007 가이드라인 준수
    # (num_leaves ≤ 31, n_estimators ≤ 200 → 모델 크기·SHAP 속도 제어)
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y, feature_name=FEATURE_COLS)

    explainer = shap.TreeExplainer(model)
    return model, explainer, medians


def predict_one(model, feature_vec: dict) -> float:
    """단일 학교/입력에 대한 TBRI 점수 추론. 학습 시 사용한 피처명 그대로 DataFrame 전달."""
    x = pd.DataFrame([{c: float(feature_vec[c]) for c in FEATURE_COLS}], columns=FEATURE_COLS)
    return float(model.predict(x)[0])


def shap_for_one(explainer, feature_vec: dict) -> np.ndarray:
    """단일 입력의 SHAP 값 배열 반환 (길이=피처 수)."""
    x = pd.DataFrame([{c: float(feature_vec[c]) for c in FEATURE_COLS}], columns=FEATURE_COLS)
    sv = explainer.shap_values(x)
    # LightGBM 회귀는 (1, n_features) 형태
    return sv[0] if sv.ndim == 2 else sv


# ─────────────────────────────────────────────────────────────────────────────
# 시각화 컴포넌트
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_bar(shap_values: np.ndarray, feature_vec: dict, top_k: int = 5) -> go.Figure:
    """SHAP 기여도를 Plotly 가로 막대로 시각화 (절대값 상위 top_k)."""
    abs_sv = np.abs(shap_values)
    order = np.argsort(abs_sv)[::-1][:top_k]

    selected_sv = shap_values[order]
    selected_names = [FEATURE_LABELS_KOR[FEATURE_COLS[i]] for i in order]
    selected_vals = [feature_vec[FEATURE_COLS[i]] for i in order]

    # 양수(빨강)=점수 상승=위험 증가, 음수(파랑)=점수 하락=위험 완화
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in selected_sv]
    y_labels = [f"{n}<br><sub>현재 값: {v:.2f}</sub>" for n, v in zip(selected_names, selected_vals)]

    fig = go.Figure(
        go.Bar(
            x=selected_sv,
            y=y_labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in selected_sv],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"주요 위험 요인 (상위 {top_k}개) — SHAP 기여도",
        xaxis_title="TBRI 점수에 대한 기여 (양수=위험↑, 음수=위험↓)",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=10, r=30, t=60, b=40),
    )
    return fig


def build_region_map(df: pd.DataFrame) -> folium.Map:
    """지역(시군구) 단위 평균 TBRI를 Folium 원형 마커로 표시."""
    agg = (
        df.dropna(subset=["위도", "경도"])
        .groupby("익명_지역", as_index=False)
        .agg(
            avg_tbri=("tbri_score", "mean"),
            n_schools=("정보공시 학교코드", "count"),
            lat=("위도", "mean"),
            lon=("경도", "mean"),
        )
    )
    if agg.empty:
        # 위·경도 결측이 전체일 경우 안전한 기본 지도
        m = folium.Map(location=[37.5, 127.0], zoom_start=8, tiles="cartodbpositron")
        return m

    m = folium.Map(
        location=[agg["lat"].mean(), agg["lon"].mean()],
        zoom_start=9,
        tiles="cartodbpositron",
    )
    for _, row in agg.iterrows():
        color = risk_color(row["avg_tbri"])
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=float(8 + np.log1p(row["n_schools"]) * 2),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(
                (
                    f"<b>{row['익명_지역']}</b><br>"
                    f"평균 TBRI: {row['avg_tbri']:.1f}<br>"
                    f"학교 수: {int(row['n_schools'])}"
                ),
                max_width=220,
            ),
            tooltip=f"{row['익명_지역']} — 평균 TBRI {row['avg_tbri']:.1f}",
        ).add_to(m)
    return m


def build_school_map(df_subset: pd.DataFrame) -> folium.Map:
    """선택한 지역 내 학교별 위험도를 표시 (익명 라벨만 사용)."""
    df_subset = df_subset.dropna(subset=["위도", "경도"]).copy()
    if df_subset.empty:
        return folium.Map(location=[37.5, 127.0], zoom_start=9, tiles="cartodbpositron")

    m = folium.Map(
        location=[df_subset["위도"].mean(), df_subset["경도"].mean()],
        zoom_start=11,
        tiles="cartodbpositron",
    )
    for _, row in df_subset.iterrows():
        color = risk_color(row["tbri_score"])
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1,
            tooltip=f"{row['익명_학교명']} — TBRI {row['tbri_score']:.1f} ({row['risk_grade']})",
        ).add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# UI 페이지
# ─────────────────────────────────────────────────────────────────────────────

def page_admin(df: pd.DataFrame, model, explainer, medians: dict):
    """관리자 대시보드: 위험 지도 / 학교 리포트 / What-if 시뮬레이터."""
    st.header("🛡️ 관리자 대시보드")
    st.caption(
        "교육청·교원치유지원센터 운영자용. 모든 학교명·지역명은 익명 라벨로 표시됩니다. "
        f"기준 연도: {TARGET_YEAR} · 분석 대상: {len(df):,}개교 (경기도교육청)"
    )

    tab_map, tab_report, tab_whatif = st.tabs(
        ["📍 지역별 위험 지도", "🏫 학교별 TBRI 리포트", "🧪 What-if 시뮬레이터"]
    )

    # ── Tab 1: 지역별 위험 지도
    with tab_map:
        st.subheader("경기도 지역별 평균 TBRI")
        st.caption("원의 색=평균 위험 등급, 크기=학교 수")

        col_l, col_r = st.columns([3, 1])
        with col_l:
            region_map = build_region_map(df)
            st_folium(region_map, height=520, width=None, returned_objects=[])
        with col_r:
            st.markdown("**위험 등급**")
            st.markdown("🟢 녹색: 0~33 (저위험)")
            st.markdown("🟠 황색: 34~66 (주의)")
            st.markdown("🔴 적색: 67~100 (고위험)")

            grade_counts = df["risk_grade"].value_counts()
            st.markdown("**전체 학교 위험 분포**")
            for g in ["녹색(저위험)", "황색(주의)", "적색(고위험)"]:
                st.write(f"- {g}: {int(grade_counts.get(g, 0)):,}개교")

        st.divider()
        st.subheader("지역 선택 후 학교 단위 보기")
        regions = sorted(df["익명_지역"].dropna().unique())
        sel_region = st.selectbox("지역 선택", regions, key="admin_region")
        sub = df[df["익명_지역"] == sel_region]
        st.write(
            f"{sel_region} 내 {len(sub):,}개교 — 평균 TBRI {sub['tbri_score'].mean():.1f}"
        )
        st_folium(build_school_map(sub), height=420, width=None, returned_objects=[])

    # ── Tab 2: 학교별 TBRI 리포트
    with tab_report:
        st.subheader("학교별 TBRI 상세 리포트")
        # 학교 선택을 좁히기 위해 지역 필터 먼저
        col1, col2 = st.columns(2)
        with col1:
            sel_region2 = st.selectbox(
                "지역 선택", sorted(df["익명_지역"].dropna().unique()), key="report_region"
            )
        with col2:
            sub_school = df[df["익명_지역"] == sel_region2]
            sel_school = st.selectbox(
                "학교 선택", sorted(sub_school["익명_학교명"].unique()), key="report_school"
            )

        row = df[df["익명_학교명"] == sel_school].iloc[0]
        feat_vec = {c: float(row[c]) for c in FEATURE_COLS}
        tbri = predict_one(model, feat_vec)
        sv = shap_for_one(explainer, feat_vec)

        m1, m2, m3 = st.columns(3)
        m1.metric("TBRI 점수", f"{tbri:.1f} / 100")
        m2.metric("위험 등급", risk_grade(tbri))
        m3.metric("위험 등급(저장값)", row["risk_grade"])

        st.plotly_chart(plot_shap_bar(sv, feat_vec, top_k=5), width="stretch")

        # 자연어 설명 (상위 3개)
        st.markdown("#### 위험 요인 해설")
        order = np.argsort(np.abs(sv))[::-1][:3]
        for idx in order:
            feat = FEATURE_COLS[idx]
            label = FEATURE_LABELS_KOR[feat]
            direction = "위험을 높이는" if sv[idx] > 0 else "위험을 낮추는"
            st.markdown(
                f"- **{label}** (현재 값 {feat_vec[feat]:.2f}, SHAP {sv[idx]:+.2f}) — "
                f"이 요인은 현재 {direction} 방향으로 작용합니다. "
                f"{RISK_NARRATIVE.get(feat, '')}"
            )

    # ── Tab 3: What-if 시뮬레이터
    with tab_whatif:
        st.subheader("What-if 시뮬레이터")
        st.caption("학교 조건을 변경했을 때 TBRI가 어떻게 변하는지 즉시 확인합니다.")

        col1, col2 = st.columns(2)
        with col1:
            sel_region3 = st.selectbox(
                "지역 선택", sorted(df["익명_지역"].dropna().unique()), key="wi_region"
            )
        with col2:
            sub_w = df[df["익명_지역"] == sel_region3]
            sel_school3 = st.selectbox(
                "기준 학교", sorted(sub_w["익명_학교명"].unique()), key="wi_school"
            )

        base_row = df[df["익명_학교명"] == sel_school3].iloc[0]
        base_vec = {c: float(base_row[c]) for c in FEATURE_COLS}
        base_tbri = predict_one(model, base_vec)

        st.markdown(f"**기준 TBRI: {base_tbri:.1f} ({risk_grade(base_tbri)})**")
        st.markdown("##### 조건 조정")
        sim_vec = dict(base_vec)
        cols = st.columns(2)
        with cols[0]:
            sim_vec["학급당 학생수(계)"] = st.slider(
                "학급당 학생수", 0.0, 40.0, float(base_vec["학급당 학생수(계)"]), 0.5, key="wi_cls"
            )
            sim_vec["수업교원 1인당 학생수"] = st.slider(
                "교사 1인당 학생수", 0.0, 50.0, float(base_vec["수업교원 1인당 학생수"]), 0.5, key="wi_tps"
            )
            sim_vec["주당평균수업시수(교사 1인당)"] = st.slider(
                "주당 평균 수업시수", 0.0, 30.0,
                float(base_vec["주당평균수업시수(교사 1인당)"]), 0.1, key="wi_hrs"
            )
        with cols[1]:
            sim_vec["기간제_비율"] = st.slider(
                "기간제 교사 비율", 0.0, 1.0, float(base_vec["기간제_비율"]), 0.01, key="wi_inter"
            )
            sim_vec["휴직_비율"] = st.slider(
                "교원 휴직률", 0.0, 0.5, float(base_vec["휴직_비율"]), 0.01, key="wi_leave"
            )
            sim_vec["정교사_비율"] = st.slider(
                "정교사 비율", 0.0, 1.0, float(base_vec["정교사_비율"]), 0.01, key="wi_reg"
            )

        new_tbri = predict_one(model, sim_vec)
        delta = new_tbri - base_tbri
        c1, c2, c3 = st.columns(3)
        c1.metric("조정 후 TBRI", f"{new_tbri:.1f}", delta=f"{delta:+.1f}")
        c2.metric("조정 후 위험 등급", risk_grade(new_tbri))
        c3.metric("기준 대비 변화 방향", "위험 상승" if delta > 0 else ("위험 완화" if delta < 0 else "변화 없음"))


def page_self_check(model, explainer, medians: dict):
    """교사 셀프체크: 5개 입력 → TBRI/SHAP/지원 안내. 익명·로그인 없음."""
    st.header("👩‍🏫 교사 셀프체크 (익명)")
    st.caption(
        "어떠한 개인정보(이름·연락처·기기 식별자·접속 기록)도 수집하거나 저장하지 않습니다. "
        "입력값은 세션 종료 시 즉시 폐기됩니다."
    )

    st.markdown("##### 우리 학교 조건 입력 (5개 항목)")
    col1, col2 = st.columns(2)
    with col1:
        cls_size = st.slider(
            "1️⃣ 학급당 학생수", 0, 40, int(medians["학급당 학생수(계)"]), 1
        )
        teacher_load = st.slider(
            "2️⃣ 교사 1인당 학생수", 0, 50, int(medians["수업교원 1인당 학생수"]), 1
        )
        weekly_hours = st.slider(
            "3️⃣ 주당 평균 수업시수", 0.0, 30.0,
            float(medians["주당평균수업시수(교사 1인당)"]), 0.1
        )
    with col2:
        interim_ratio = st.slider(
            "4️⃣ 기간제 교사 비율", 0.0, 1.0,
            float(medians["기간제_비율"]), 0.01
        )
        wee_installed = st.radio(
            "5️⃣ 학교 내 WEE클래스 설치 여부",
            options=["설치됨", "미설치"],
            index=1 if medians["WEE_미설치"] >= 0.5 else 0,
            horizontal=True,
        )

    if st.button("🔎 TBRI 결과 확인", type="primary"):
        # 5개 외 피처는 전국 중앙값(imputation)으로 보정 — 셀프체크 정확도 한계 명시
        vec = dict(medians)
        vec["학급당 학생수(계)"] = float(cls_size)
        vec["수업교원 1인당 학생수"] = float(teacher_load)
        vec["주당평균수업시수(교사 1인당)"] = float(weekly_hours)
        vec["기간제_비율"] = float(interim_ratio)
        vec["WEE_미설치"] = 1.0 if wee_installed == "미설치" else 0.0

        tbri = predict_one(model, vec)
        sv = shap_for_one(explainer, vec)

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("TBRI 점수", f"{tbri:.1f} / 100")
            st.metric("위험 등급", risk_grade(tbri))
            st.caption(
                "※ 입력하지 않은 항목은 경기도 전체 중앙값으로 자동 보정된 근사 추정값입니다."
            )
        with c2:
            st.plotly_chart(plot_shap_bar(sv, vec, top_k=5), width="stretch")

        st.markdown("#### 위험 요인 해설 (상위 3개)")
        order = np.argsort(np.abs(sv))[::-1][:3]
        for idx in order:
            feat = FEATURE_COLS[idx]
            label = FEATURE_LABELS_KOR[feat]
            direction = "위험을 높이는" if sv[idx] > 0 else "위험을 낮추는"
            st.markdown(
                f"- **{label}** — 현재 {direction} 방향으로 작용. {RISK_NARRATIVE.get(feat, '')}"
            )

        st.markdown("---")
        st.markdown("#### 🤝 맞춤 지원 안내")
        if tbri >= 67:
            st.error(
                "**고위험 환경**으로 판정되었습니다. 다음 지원을 권장합니다.\n"
                "- 교원치유지원센터 무료 상담 (지역 교육청 안내)\n"
                "- 학교 단위 업무경감 컨설팅 신청\n"
                "- 동료 멘토링·공동체 회복 프로그램"
            )
        elif tbri >= 34:
            st.warning(
                "**주의 수준**입니다. 예방적 자기관리·동료 지지 활동을 권장합니다.\n"
                "- 시도교육청 교원 힐링 캠프\n"
                "- 마음건강 자가검진 (한국교원공제회 등 안내 자료)"
            )
        else:
            st.success(
                "**저위험 환경**입니다. 현재의 건강한 환경을 유지하기 위한 자기 돌봄을 권장합니다."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 앱 엔트리
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🍃",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🍃 {APP_TITLE}")
    st.markdown(f"##### {APP_SUBTITLE}")

    # 사이드바: 메뉴 라우팅
    with st.sidebar:
        st.markdown(f"## {APP_TITLE}")
        st.caption("제8회 교육 공공데이터 AI 활용대회 출품작")
        page = st.radio(
            "메뉴",
            ["관리자 대시보드", "교사 셀프체크"],
            index=0,
            key="page_selector",
        )
        st.divider()
        st.markdown("### 📊 데이터")
        st.markdown(f"- 출처: 경기도교육청 공시 7종")
        st.markdown(f"- 기준 연도: **{TARGET_YEAR}**")
        st.markdown("### 🤖 모델")
        st.markdown("- LightGBM Regressor")
        st.markdown("- SHAP TreeExplainer")
        st.divider()
        st.caption(
            "🔒 어떠한 개인정보도 수집·저장하지 않습니다. "
            "모든 학교·지역명은 익명 라벨로 표시됩니다."
        )

    # 데이터·모델 로드 (캐시됨)
    df, medians = get_master_with_scores(TARGET_YEAR)
    model, explainer, medians_from_model = get_model_and_explainer()
    # medians는 동일하지만 명시적으로 한 군데에서 사용

    if page == "관리자 대시보드":
        page_admin(df, model, explainer, medians)
    else:
        page_self_check(model, explainer, medians)


if __name__ == "__main__":
    main()
