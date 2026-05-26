# =============================================================================
# 쉼표(Swimpyo) — AI 기반 교사 업무·감정 부담 조기경보 시스템 (프로토타입)
# 제8회 교육 공공데이터 AI 활용대회 출품작
# =============================================================================
#
# [아키텍처]  모듈러 모놀리스 + 파일 기반 산출물 (docs/acad/architecture.md, ADR-001)
#   본 단일 파일은 ACAD 설계의 논리 모듈을 섹션으로 반영한다:
#     [common]  설정·표시 컬럼·유틸
#     [etl]     공공데이터(CSV) 로드·전처리·피처 엔지니어링  (오프라인 배치 역할)
#     [model]   LightGBM 학습·TBRI 추론·SHAP 설명             (인메모리)
#     [support] 위험요인 → 지원자원 룰 매핑                    (LLM 비사용, 룰 기반)
#     [admin]   관리자 대시보드 (위험지도 / 학교리포트 / What-if)
#     [teacher] 교사 셀프체크 웹앱
#
# [데이터]  경기도교육청 학교알리미 공시데이터 (dataset/ 폴더, UTF-8-sig, 2022~2025)
#   - 직위별 교원 현황          → 타깃(휴직비율) + 기간제/보직/여교원 비율
#   - 수업일수 및 수업시수 현황 → 교사 1인당 주당 평균 수업시수 (업무 부담)
#   - 학년별 학급별 학생수      → 학급당 학생수, 교원 1인당 학생수 (업무 부담)
#   - 학생 학부모 상담 현황     → 내부 상담 실적, WEE클래스 설치 (정서노동·지원자원)
#   - 동아리 활동 현황          → 교사당 동아리 지도 부담 (추가 업무)
#   - 학교기본정보              → 위도·경도(지도), 설립구분, 지역, 학교급
#   ※ 6종 모두 '정보공시 학교코드' 기준으로 (학교코드, 연도) 패널 조인 (조인율 100%)
#
# [모델 선택 이유]  LightGBM (ADR-005)
#   - 테이블형(정형) 데이터에 최적인 Gradient Boosting 계열
#   - XGBoost 대비 메모리 효율이 높아 Streamlit Cloud 무료 티어(~1GB)에 적합
#   - 학습/추론이 빠르고 SHAP TreeExplainer를 공식 지원 → 설명가능 AI 구현 용이
#   - 검증 결과: 휴직비율 회귀 R²≈0.51 / 고위험(상위25%) 분류 AUC≈0.86
#
# [TBRI 정의]  Teacher Burnout Risk Index — 학교 '환경'의 구조적 위험도(0~100)
#   - 모델이 예측한 '교원 휴직비율'을 전체 학교 분포의 백분위로 환산
#   - ※ 특정 교사 개인의 상태가 아니라 학교 단위 구조적 위험을 나타냄 (프라이버시 원칙)
#
# [데이터 공개성]  학교알리미 공시데이터는 공개 정보이므로 학교명·지역명을 그대로 사용
#   - 교사 셀프체크는 개인 식별 정보를 미수집·미저장 (DB 미사용, 세션 휘발성)
#   - ※ 대회 제출 문서(제안서·PDF)에는 학교명·지역명 비공개 규정이 적용됨에 유의
# =============================================================================

import os
import glob

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu

# =============================================================================
# [common] 설정·상수
# =============================================================================
DATASET_DIR = "dataset"

# 카테고리 폴더명 (dataset/ 하위)
CAT_DIRS = {
    "직위별": "직위별 교원 현황",
    "수업시수": "수업일수 및 수업시수 현황",
    "학급학생": "학년별 학급별 학생수",
    "상담": "학생 학부모 상담계획 및 실시현황",
    "동아리": "동아리 활동현황",
    "학교기본": "학교기본정보",
}

KEY = "정보공시 학교코드"

# 모델 입력 피처 (학습·추론 시 항상 이 순서를 사용)
FEATURE_COLS = [
    "총교원", "기간제비율", "보직비율", "여교원비율",
    "주당평균수업시수", "학급당학생수", "교원1인당학생수", "총학생수",
    "특수학급수", "내부상담건수", "WEE설치", "정규동아리수", "교사당동아리수",
    "학교급코드", "설립코드",
]

# 화면 표기용 한글 라벨 (SHAP 차트 등)
FEATURE_LABELS = {
    "총교원": "전체 교원 수",
    "기간제비율": "기간제 교사 비율",
    "보직비율": "보직 교사 비율",
    "여교원비율": "여성 교원 비율",
    "주당평균수업시수": "교사 1인당 주당 수업시수",
    "학급당학생수": "학급당 학생 수",
    "교원1인당학생수": "교원 1인당 학생 수",
    "총학생수": "전체 학생 수",
    "특수학급수": "특수학급 수",
    "내부상담건수": "내부 상담 실적(건)",
    "WEE설치": "WEE클래스 설치 여부",
    "정규동아리수": "정규 동아리 수",
    "교사당동아리수": "교사 1인당 동아리 지도 수",
    "학교급코드": "학교급(초/중/고)",
    "설립코드": "설립구분(공/사립)",
}

LEVEL_NAME = {0: "초등학교", 1: "중학교", 2: "고등학교"}
GRADE_COLOR = {"녹색": "#2e9e5b", "황색": "#e8a33d", "적색": "#d6453d"}
GRADE_DESC = {"녹색": "양호", "황색": "주의", "적색": "위험"}


def _num(s):
    """문자열 컬럼을 안전하게 숫자로 변환 (변환 불가 값은 NaN)."""
    return pd.to_numeric(s, errors="coerce")


# =============================================================================
# [common] 표시용 컬럼 준비
#   - 학교알리미 공시데이터는 공개 정보이므로 실제 학교명·지역명을 그대로 사용한다.
# =============================================================================
def prepare_display(df: pd.DataFrame) -> pd.DataFrame:
    """화면 표시용 '학교' 컬럼을 실제 학교명으로 설정한다 (익명화하지 않음)."""
    df = df.copy()
    df["학교"] = df["학교명"]
    return df


# =============================================================================
# [etl] 공공데이터 로드 + 전처리 + 피처 엔지니어링
#   - @st.cache_data: CSV 로드·병합 결과를 캐싱하여 재실행 시 즉시 반환 (AD-U001)
# =============================================================================
def _load_category(folder: str) -> pd.DataFrame:
    """카테고리 폴더 내 모든 CSV를 읽어 연도·학교급 컬럼을 붙여 결합.
    파일명 규칙: '2024년도_...(고)_경기도교육청.csv' → 연도=2024, 학교급=고."""
    frames = []
    for fp in glob.glob(os.path.join(DATASET_DIR, folder, "*.csv")):
        bn = os.path.basename(fp)
        # 학교알리미 공시 CSV는 UTF-8 BOM(utf-8-sig) 인코딩
        d = pd.read_csv(fp, encoding="utf-8-sig", dtype=str)
        d["연도"] = bn[:4] if bn[:4].isdigit() else "기준"
        d["학교급"] = "초" if "(초)" in bn else "중" if "(중)" in bn else "고" if "(고)" in bn else "기타"
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="공공데이터를 불러오고 전처리하는 중…")
def load_features() -> pd.DataFrame:
    """6종 공시데이터를 (학교코드, 연도) 기준으로 병합하고 모델 피처를 생성한다.

    전처리 로직 요약:
      1) 직위별 교원 현황: 휴직비율(타깃) = 총계(휴직교원)/총계(계),
         기간제·보직·여교원 비율을 총교원 대비 비율로 파생
      2) 수업시수: 교사 1인당 주당 평균 수업시수 (업무 강도)
      3) 학급학생: 학급당 학생수, 교원 1인당 학생수, 총학생수, 특수학급수
      4) 상담: 내부 상담 실적 건수, WEE클래스 설치여부(0/1)
      5) 동아리: 정규 동아리수, 교사 1인당 동아리 지도수(=동아리수/지도교사수)
      6) 학교기본정보: 위도·경도, 설립구분, 지역(연도 무관 → 학교코드로만 조인)
      7) 결측치: 비율·건수형은 학교급별 중앙값으로 대치(견고한 통계량, 이상치에 둔감)
    """
    jik = _load_category(CAT_DIRS["직위별"])
    su = _load_category(CAT_DIRS["수업시수"])
    hak = _load_category(CAT_DIRS["학급학생"])
    sang = _load_category(CAT_DIRS["상담"])
    dong = _load_category(CAT_DIRS["동아리"])
    gibon = _load_category(CAT_DIRS["학교기본"])

    # 1) 직위별 → 타깃 + 인력 구성 비율
    jik["총교원"] = _num(jik["총계(계)"])
    jik["휴직비율"] = _num(jik["총계(휴직교원)"]) / jik["총교원"]
    jik["기간제비율"] = _num(jik["기간제교사(계)"]) / jik["총교원"]
    jik["보직비율"] = _num(jik["보직교사(계)"]) / jik["총교원"]
    jik["여교원비율"] = _num(jik["총계(여)"]) / jik["총교원"]
    base = jik[[KEY, "학교명", "연도", "학교급", "총교원", "휴직비율",
                "기간제비율", "보직비율", "여교원비율"]].copy()

    # 2) 수업시수
    su["주당평균수업시수"] = _num(su["주당평균수업시수(교사 1인당)"])
    base = base.merge(su[[KEY, "연도", "주당평균수업시수"]], on=[KEY, "연도"], how="left")

    # 3) 학급학생
    hak["학급당학생수"] = _num(hak["학급당 학생수(계)"])
    hak["교원1인당학생수"] = _num(hak["수업교원 1인당 학생수"])
    hak["총학생수"] = _num(hak["학생수(계)"])
    hak["특수학급수"] = _num(hak["특수학급 학급수"])
    base = base.merge(
        hak[[KEY, "연도", "학급당학생수", "교원1인당학생수", "총학생수", "특수학급수"]],
        on=[KEY, "연도"], how="left")

    # 4) 상담 (정서노동·지원자원 지표)
    sang["내부상담건수"] = _num(sang["상담실적(내부상담전문가)"])
    sang["WEE설치"] = (sang["교내 WEE클래스 설치여부"].fillna("").str.strip().str.upper() == "Y").astype(int)
    base = base.merge(sang[[KEY, "연도", "내부상담건수", "WEE설치"]], on=[KEY, "연도"], how="left")

    # 5) 동아리 (동일 헤더가 정규/자율 2세트 → pandas가 자동 리네임, 첫 세트=정규동아리 사용)
    dong["정규동아리수"] = _num(dong["동아리수"])
    dong["동아리지도교사수"] = _num(dong["지도교사수"])
    base = base.merge(dong[[KEY, "연도", "정규동아리수", "동아리지도교사수"]], on=[KEY, "연도"], how="left")
    base["교사당동아리수"] = base["정규동아리수"] / base["동아리지도교사수"].replace(0, np.nan)

    # 6) 학교기본정보 (연도 무관 스냅샷 → 학교코드로만 조인)
    gibon["위도"] = _num(gibon["위도"])
    gibon["경도"] = _num(gibon["경도"])
    g = gibon[[KEY, "지역", "설립구분", "위도", "경도"]].drop_duplicates(KEY)
    base = base.merge(g, on=KEY, how="left")

    # 범주형 인코딩
    base["학교급코드"] = base["학교급"].map({"초": 0, "중": 1, "고": 2})
    base["설립코드"] = (base["설립구분"] == "사립").astype(int)

    # 타깃 결측 행 제거 + 비정상 비율(>1) 제거
    base = base[base["휴직비율"].notna() & (base["휴직비율"] <= 1)].copy()

    # 7) 결측치 대치 — 학교급별 중앙값 (이상치에 둔감한 견고한 통계량)
    fill_cols = ["주당평균수업시수", "학급당학생수", "교원1인당학생수", "총학생수",
                 "특수학급수", "내부상담건수", "정규동아리수", "교사당동아리수",
                 "기간제비율", "보직비율", "여교원비율"]
    for c in fill_cols:
        base[c] = base.groupby("학교급코드")[c].transform(lambda s: s.fillna(s.median()))
    base["WEE설치"] = base["WEE설치"].fillna(0)
    # 위경도 결측은 지도에서만 제외 (모델 피처 아님)

    # 표시용 컬럼 준비 (공개 공시데이터 → 실제 학교명·지역명 그대로 사용)
    base = prepare_display(base)
    return base.reset_index(drop=True)


# =============================================================================
# [model] LightGBM 학습 + TBRI 추론 + SHAP
#   - @st.cache_resource: 모델·explainer 등 직렬화 불가 객체를 1회 학습 후 캐싱
# =============================================================================
@st.cache_resource(show_spinner="AI 모델을 학습하는 중…")
def train_model(df: pd.DataFrame):
    """휴직비율을 예측하는 LightGBM 회귀 모델 학습 + 평가 지표 + SHAP explainer 반환.

    - 회귀: 휴직비율(연속값) 예측 → TBRI 백분위 환산의 기준
    - 분류 지표(AUC): 상위 25%를 고위험으로 본 이진 분류 성능을 참고로 함께 산출
    - SHAP TreeExplainer: 개별 학교의 위험 요인 기여도 분해 (설명가능 AI)
    """
    X = df[FEATURE_COLS]
    y = df["휴직비율"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    pred_te = model.predict(X_te)
    r2 = r2_score(y_te, pred_te)
    mae = mean_absolute_error(y_te, pred_te)
    # 참고용 분류 AUC (상위 25% 고위험)
    thr = y.quantile(0.75)
    auc = roc_auc_score((y_te > thr).astype(int), pred_te)

    # TBRI 백분위 환산 기준: 전체 데이터 예측값 분포(정렬)
    ref_pred = np.sort(model.predict(X))

    # 학교급별 피처 중앙값 (교사 셀프체크에서 미입력 피처 채움용)
    median_by_level = {
        lvl: df[df["학교급코드"] == lvl][FEATURE_COLS].median().to_dict()
        for lvl in df["학교급코드"].unique()
    }

    explainer = shap.TreeExplainer(model)
    metrics = {"r2": r2, "mae": mae, "auc": auc, "n": len(df)}
    return model, ref_pred, explainer, median_by_level, metrics


def to_tbri(pred_value: float, ref_pred: np.ndarray) -> float:
    """예측 휴직비율 → 전체 분포 백분위(0~100) = TBRI 점수."""
    rank = np.searchsorted(ref_pred, pred_value) / len(ref_pred)
    return float(np.clip(rank * 100, 0, 100))


def tbri_grade(tbri: float) -> str:
    """TBRI 점수 → 3단계 등급. 백분위 50/75 임계값 기준."""
    if tbri >= 75:
        return "적색"
    if tbri >= 50:
        return "황색"
    return "녹색"


def add_tbri_columns(df: pd.DataFrame, model, ref_pred: np.ndarray) -> pd.DataFrame:
    """데이터프레임 전체에 예측·TBRI·등급 컬럼 추가 (지도/리포트용)."""
    df = df.copy()
    df["예측휴직비율"] = model.predict(df[FEATURE_COLS])
    df["TBRI"] = [to_tbri(p, ref_pred) for p in df["예측휴직비율"]]
    df["등급"] = df["TBRI"].apply(tbri_grade)
    return df


def shap_top_factors(explainer, feature_row: pd.DataFrame, top_n: int = 6):
    """단일 학교(또는 셀프체크 입력)의 SHAP 기여도 상위 요인을 반환.
    반환: [(라벨, shap값), ...] shap값 부호 = 위험 증가(+)/감소(-)."""
    sv = explainer.shap_values(feature_row[FEATURE_COLS])
    sv = np.array(sv).reshape(-1)
    contributions = sorted(
        zip(FEATURE_COLS, sv), key=lambda t: abs(t[1]), reverse=True
    )[:top_n]
    return [(FEATURE_LABELS[f], float(v)) for f, v in contributions]


def plot_shap_bar(factors):
    """SHAP 기여도를 Plotly 수평 막대로 시각화 (위험↑ 빨강, 위험↓ 파랑)."""
    labels = [f[0] for f in factors][::-1]
    values = [f[1] for f in factors][::-1]
    colors = ["#d6453d" if v > 0 else "#3d7fd6" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="위험 요인별 기여도 (SHAP) — 오른쪽(+)일수록 위험을 높임",
        xaxis_title="TBRI(휴직비율)에 대한 기여도", yaxis_title="",
        height=340, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_gauge(tbri: float, grade: str):
    """TBRI 점수 게이지 차트 (Plotly)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(tbri, 1),
        number={"suffix": " 점"},
        title={"text": f"TBRI — {GRADE_DESC[grade]}({grade})"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": GRADE_COLOR[grade]},
            "steps": [
                {"range": [0, 50], "color": "#e8f5ec"},
                {"range": [50, 75], "color": "#fdf3e3"},
                {"range": [75, 100], "color": "#fbe9e7"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# =============================================================================
# [support] 위험 요인 → 지원 자원 룰 매핑 (LLM 비사용, 룰 기반 — ADR-003 폴백 정신)
# =============================================================================
SUPPORT_RULES = {
    "교사 1인당 주당 수업시수": [
        "수업시수 재배분·교과 전담 인력 보강 검토",
        "교육지원청 강사풀·기간제 배치 우선순위 상향",
    ],
    "교원 1인당 학생 수": [
        "과밀 해소를 위한 학급 증설·교원 정원 재산정 건의",
        "행정업무 경감(업무지원팀) 우선 적용 대상 지정",
    ],
    "학급당 학생 수": [
        "과밀학급 우선 지원 대상 지정",
        "학급당 학생 수 상한 정책 연계",
    ],
    "기간제 교사 비율": [
        "정규 교원 정원 확보 건의로 고용 안정성 개선",
        "기간제 교사 대상 멘토링·적응 지원 프로그램",
    ],
    "내부 상담 실적(건)": [
        "교원 정서 지원: 교원치유지원센터 집단상담 연계",
        "WEE센터 외부 상담전문가 파견 확대",
    ],
    "교사 1인당 동아리 지도 수": [
        "동아리 지도 외부강사·자원봉사 매칭 확대",
        "방과후·동아리 운영 행정부담 경감",
    ],
    "보직 교사 비율": [
        "보직 업무 분장 재조정·업무경감 컨설팅",
    ],
}
DEFAULT_SUPPORT = [
    "교원치유지원센터 일반 상담·힐링 프로그램 안내",
    "교육지원청 교원복지 담당 연계",
]


def match_support(top_factors):
    """SHAP 상위 '위험 증가(+)' 요인에 대응하는 지원 프로그램 목록 반환."""
    recs = []
    for label, val in top_factors:
        if val > 0 and label in SUPPORT_RULES:
            recs.extend(SUPPORT_RULES[label])
    if not recs:
        recs = DEFAULT_SUPPORT
    # 중복 제거(순서 유지)
    seen, out = set(), []
    for r in recs:
        if r not in seen:
            seen.add(r); out.append(r)
    return out[:5]


# =============================================================================
# [admin] 관리자 대시보드 페이지
# =============================================================================
def page_admin_map(df_t: pd.DataFrame):
    st.subheader("전국 위험 지도 — 경기도 학교 TBRI")
    st.caption("경기도교육청 학교알리미 공개 공시데이터(위경도·학교명) 기준입니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        lvl = st.selectbox("학교급", [2, 1, 0], format_func=lambda x: LEVEL_NAME[x])
    with c2:
        years = sorted(df_t["연도"].unique())
        year = st.selectbox("연도", years, index=len(years) - 1)
    with c3:
        # 등급 필터 — 멀티셀렉트(모두 같은 색) 대신 등급별 색상 체크박스로 구분
        st.markdown("**등급 필터**")
        show_red = st.checkbox("🟥 적색 (위험)", value=True)
        show_org = st.checkbox("🟧 황색 (주의)", value=True)
        show_grn = st.checkbox("🟩 녹색 (양호)", value=True)
    grades = [g for g, on in
              [("적색", show_red), ("황색", show_org), ("녹색", show_grn)] if on]

    sub = df_t[(df_t["학교급코드"] == lvl) & (df_t["연도"] == year)
               & (df_t["등급"].isin(grades))].copy()
    sub = sub.dropna(subset=["위도", "경도"])

    # 등급 분포 요약
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("대상 학교", f"{len(sub):,}곳")
    m2.metric("적색(위험)", f"{(sub['등급']=='적색').sum():,}곳")
    m3.metric("황색(주의)", f"{(sub['등급']=='황색').sum():,}곳")
    m4.metric("평균 TBRI", f"{sub['TBRI'].mean():.1f}" if len(sub) else "-")

    if len(sub) == 0:
        st.info("선택 조건에 해당하는 학교가 없습니다.")
        return

    # Folium 지도 — 경기도 중심, 학교별 CircleMarker (등급 색상)
    center = [sub["위도"].mean(), sub["경도"].mean()]
    fmap = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(fmap)
    for _, r in sub.iterrows():
        folium.CircleMarker(
            location=[r["위도"], r["경도"]],
            radius=5, color=GRADE_COLOR[r["등급"]], fill=True,
            fill_color=GRADE_COLOR[r["등급"]], fill_opacity=0.8,
            tooltip=f"{r['학교']} · {r['지역']} · TBRI {r['TBRI']:.0f}({r['등급']})",
        ).add_to(cluster)
    st_folium(fmap, height=460, use_container_width=True, returned_objects=[])

    # 고위험 상위 표
    st.markdown("##### 고위험 학교 TOP 15")
    top = sub.sort_values("TBRI", ascending=False).head(15)
    st.dataframe(
        top[["학교", "지역", "TBRI", "등급", "주당평균수업시수",
             "교원1인당학생수", "기간제비율"]].round(2),
        width="stretch", hide_index=True,
    )


def page_admin_report(df_t: pd.DataFrame, explainer):
    st.subheader("학교별 TBRI 상세 리포트")
    c1, c2, c3 = st.columns(3)
    with c1:
        lvl = st.selectbox("학교급", [2, 1, 0], format_func=lambda x: LEVEL_NAME[x], key="rep_lvl")
    with c2:
        years = sorted(df_t["연도"].unique())
        year = st.selectbox("연도", years, index=len(years) - 1, key="rep_year")
    sub = df_t[(df_t["학교급코드"] == lvl) & (df_t["연도"] == year)].copy()
    sub = sub.sort_values("TBRI", ascending=False)
    with c3:
        school = st.selectbox("학교", sub["학교"].tolist(), key="rep_school")

    row = sub[sub["학교"] == school].iloc[[0]]
    grade = row["등급"].iloc[0]
    tbri = row["TBRI"].iloc[0]

    left, right = st.columns([1, 1.3])
    with left:
        st.plotly_chart(plot_gauge(tbri, grade), width="stretch", key="gauge_report")
        st.metric("소속 지역", row["지역"].iloc[0])
        # 동일 학교급 내 백분위
        pct = (sub["TBRI"] < tbri).mean() * 100
        st.caption(f"같은 {LEVEL_NAME[lvl]} 중 위험도 상위 {100 - pct:.0f}% 수준")
    with right:
        factors = shap_top_factors(explainer, row)
        st.plotly_chart(plot_shap_bar(factors), width="stretch", key="shap_report")

    st.markdown("##### 주요 지표")
    cols = st.columns(4)
    show = [("교사 1인당 수업시수", "주당평균수업시수"), ("교원1인당 학생수", "교원1인당학생수"),
            ("학급당 학생수", "학급당학생수"), ("기간제 비율", "기간제비율")]
    for col, (label, key) in zip(cols, show):
        val = row[key].iloc[0]
        col.metric(label, f"{val:.2f}" if key == "기간제비율" else f"{val:.1f}")

    # 지원 자원 매칭 (A4)
    st.markdown("##### 지원 자원 매칭 (위험 요인 기반)")
    for rec in match_support(factors):
        st.markdown(f"- {rec}")


def page_admin_whatif(df_t: pd.DataFrame, model, ref_pred, explainer):
    st.subheader("What-if 시뮬레이터")
    st.caption("조건을 바꿔 TBRI 변화를 즉시 확인합니다. (예: 기간제 교사 비율을 낮추면?)")

    c1, c2, c3 = st.columns(3)
    with c1:
        lvl = st.selectbox("학교급", [2, 1, 0], format_func=lambda x: LEVEL_NAME[x], key="wi_lvl")
    with c2:
        years = sorted(df_t["연도"].unique())
        year = st.selectbox("연도", years, index=len(years) - 1, key="wi_year")
    sub = df_t[(df_t["학교급코드"] == lvl) & (df_t["연도"] == year)].sort_values("TBRI", ascending=False)
    with c3:
        school = st.selectbox("기준 학교", sub["학교"].tolist(), key="wi_school")

    base_row = sub[sub["학교"] == school].iloc[0]

    st.markdown("##### 조건 조정")
    s1, s2 = st.columns(2)
    with s1:
        v_susi = st.slider("교사 1인당 주당 수업시수", 5.0, 30.0,
                           float(base_row["주당평균수업시수"]), 0.5)
        v_stud = st.slider("교원 1인당 학생 수", 3.0, 30.0,
                           float(base_row["교원1인당학생수"]), 0.5)
    with s2:
        v_gigan = st.slider("기간제 교사 비율", 0.0, 0.6,
                            float(base_row["기간제비율"]), 0.01)
        v_class = st.slider("학급당 학생 수", 5.0, 40.0,
                            float(base_row["학급당학생수"]), 0.5)

    # 기준 행을 복사해 조정값만 덮어쓴 시나리오 생성
    scenario = base_row[FEATURE_COLS].copy()
    scenario["주당평균수업시수"] = v_susi
    scenario["교원1인당학생수"] = v_stud
    scenario["기간제비율"] = v_gigan
    scenario["학급당학생수"] = v_class
    scen_df = pd.DataFrame([scenario])[FEATURE_COLS]

    new_pred = float(model.predict(scen_df)[0])
    new_tbri = to_tbri(new_pred, ref_pred)
    new_grade = tbri_grade(new_tbri)
    delta = new_tbri - base_row["TBRI"]

    a, b = st.columns(2)
    a.metric("기준 TBRI", f"{base_row['TBRI']:.1f}", help=f"등급 {base_row['등급']}")
    b.metric("시뮬레이션 TBRI", f"{new_tbri:.1f}", delta=f"{delta:+.1f}",
             delta_color="inverse", help=f"등급 {new_grade}")
    st.plotly_chart(plot_gauge(new_tbri, new_grade), width="stretch", key="gauge_whatif")


# =============================================================================
# [teacher] 교사 셀프체크 웹앱
# =============================================================================
def page_teacher_selfcheck(df: pd.DataFrame, model, ref_pred, explainer, median_by_level):
    st.subheader("교사 셀프체크 — 우리 학교 환경 위험도 진단")
    st.caption("로그인·개인정보 입력이 없습니다. 입력값은 저장되지 않으며 결과 확인 즉시 사라집니다.")

    with st.form("selfcheck"):
        st.markdown("##### 우리 학교 조건 (5개 항목, 약 30초)")
        c1, c2 = st.columns(2)
        with c1:
            lvl = st.selectbox("학교급", [0, 1, 2], format_func=lambda x: LEVEL_NAME[x])
            susi = st.slider("교사 1인당 주당 평균 수업시수", 5.0, 30.0, 18.0, 0.5)
            stud = st.slider("교원 1인당 학생 수", 3.0, 30.0, 12.0, 0.5)
        with c2:
            klass = st.slider("학급당 학생 수", 5.0, 40.0, 24.0, 0.5)
            gigan = st.slider("기간제 교사 비율", 0.0, 0.6, 0.12, 0.01)
        submitted = st.form_submit_button("위험도 확인하기", width="stretch")

    if not submitted:
        return

    # 미입력 피처는 해당 학교급 중앙값으로 채움 → 5개 입력만으로 추론 (AD-U001)
    feat = dict(median_by_level.get(lvl, median_by_level[list(median_by_level)[0]]))
    feat.update({
        "학교급코드": lvl, "주당평균수업시수": susi, "교원1인당학생수": stud,
        "학급당학생수": klass, "기간제비율": gigan,
    })
    row = pd.DataFrame([feat])[FEATURE_COLS]

    pred = float(model.predict(row)[0])
    tbri = to_tbri(pred, ref_pred)
    grade = tbri_grade(tbri)

    st.divider()
    left, right = st.columns([1, 1.3])
    with left:
        st.plotly_chart(plot_gauge(tbri, grade), width="stretch", key="gauge_self")
    with right:
        factors = shap_top_factors(explainer, row)
        st.plotly_chart(plot_shap_bar(factors), width="stretch", key="shap_self")

    # 룰 기반 맞춤 안내 (외부 LLM 비사용 — 폴백 템플릿)
    st.markdown("##### 우리 학교 환경의 주요 위험 요인")
    pos = [f for f in factors if f[1] > 0][:3]
    if pos:
        st.markdown("다음 요인이 위험도를 높이는 데 기여하고 있습니다:")
        for label, _ in pos:
            st.markdown(f"- **{label}**")
    else:
        st.markdown("두드러진 위험 가중 요인은 확인되지 않았습니다. 전반적으로 양호한 편입니다.")

    st.markdown("##### 맞춤 지원 안내")
    for rec in match_support(factors):
        st.markdown(f"- {rec}")
    st.info("※ TBRI는 학교 '환경'의 구조적 위험도이며, 특정 교사 개인의 상태를 진단하지 않습니다. "
            "어려움이 있다면 교원치유지원센터·EAP 상담 등 전문 채널을 이용하세요.")


# =============================================================================
# [info] 모델·데이터 정보 페이지
#   - 어떤 데이터로 학습했는지, 지표가 무엇을 의미하는지 설명 (심사 투명성)
# =============================================================================
# 피처별 의미 설명 (모델·데이터 정보 페이지용)
FEATURE_INFO = {
    "총교원": "학교 규모(인력)",
    "기간제비율": "고용 안정성 — 높을수록 부담이 정규직에 쏠림",
    "보직비율": "행정 업무 분담 정도",
    "여교원비율": "교원 성별 구성",
    "주당평균수업시수": "수업 부담 강도(교사 1인당)",
    "학급당학생수": "학급 운영 부담",
    "교원1인당학생수": "담당 학생 부담",
    "총학생수": "학교 규모(학생)",
    "특수학급수": "특수교육 지원 부담",
    "내부상담건수": "정서노동·상담 수요",
    "WEE설치": "교내 정서지원 자원 유무",
    "정규동아리수": "비교과 운영 규모",
    "교사당동아리수": "추가 업무 부담(동아리 지도)",
    "학교급코드": "학교 유형(초/중/고)",
    "설립코드": "설립 유형(공립/사립)",
}


def page_model_info(df_t: pd.DataFrame, metrics: dict):
    st.title("모델·데이터 정보")
    st.caption("이 시스템이 어떤 데이터로, 어떻게 학습하며, 각 지표가 무엇을 의미하는지 설명합니다.")

    # 1. 학습 데이터
    st.header("1. 학습 데이터")
    n_school = df_t["학교명"].nunique()
    years = sorted(df_t["연도"].unique())
    c1, c2, c3 = st.columns(3)
    c1.metric("학습 표본", f"{metrics['n']:,} 행", help="학교 × 연도 단위")
    c2.metric("대상 학교", f"{n_school:,} 개교")
    c3.metric("대상 기간", f"{years[0]}~{years[-1]}")
    st.markdown(
        "**출처**: 경기도교육청 학교알리미 공시데이터(공개 정보). "
        "초·중·고 전체 학교를 `정보공시 학교코드` 기준으로 (학교, 연도) 패널로 결합했습니다."
    )
    data_tbl = pd.DataFrame({
        "공시 데이터셋": ["직위별 교원 현황", "수업일수 및 수업시수", "학년별·학급별 학생수",
                       "학생·학부모 상담 현황", "동아리 활동 현황", "학교기본정보"],
        "추출한 변수": [
            "휴직비율(타깃), 기간제·보직·여교원 비율, 전체 교원수",
            "교사 1인당 주당 평균 수업시수",
            "학급당 학생수, 교원 1인당 학생수, 전체 학생수, 특수학급수",
            "내부 상담 실적, WEE클래스 설치 여부",
            "정규 동아리수, 교사 1인당 동아리 지도수",
            "위도·경도(지도), 설립구분, 지역, 학교급",
        ],
    })
    st.dataframe(data_tbl, width="stretch", hide_index=True)

    # 2. 예측 타깃과 TBRI
    st.header("2. 예측 타깃과 TBRI")
    st.markdown(
        "- **예측 타깃** = 교원 **휴직비율** = 휴직 교원 수 ÷ 전체 교원 수\n"
        "- **TBRI(교사 부담 위험 지수)** = 모델이 예측한 휴직비율을 전체 학교 분포의 "
        "**백분위(0–100)**로 환산한 값\n"
        "- **등급**(백분위 기준): 🟩 녹색(0–50, 양호) · 🟧 황색(50–75, 주의) · 🟥 적색(75–100, 위험)"
    )
    st.info(
        "TBRI는 학교 **환경의 구조적 위험도**이며 특정 교사 개인의 번아웃을 진단하지 않습니다. "
        "휴직에는 질병·육아·기타 사유가 섞여 있어, 순수 번아웃이 아닌 **누적 스트레스 신호**로 해석합니다."
    )

    # 3. 입력 피처
    st.header("3. 입력 피처 (15개)")
    feat_tbl = pd.DataFrame({
        "피처": [FEATURE_LABELS[c] for c in FEATURE_COLS],
        "의미": [FEATURE_INFO[c] for c in FEATURE_COLS],
    })
    st.dataframe(feat_tbl, width="stretch", hide_index=True, height=565)

    # 4. 모델
    st.header("4. 모델: LightGBM")
    st.markdown(
        "- **알고리즘**: LightGBM 회귀 (Gradient Boosting 계열)\n"
        "- **선택 이유**: 정형(테이블) 데이터에 강하고, XGBoost보다 메모리 효율이 높아 "
        "무료 호스팅(약 1GB)에 적합하며, SHAP TreeExplainer를 공식 지원해 설명가능 AI 구현이 쉽습니다.\n"
        "- **설명가능 AI(SHAP)**: 학교별 예측에 각 요인이 얼마나 기여했는지 분해해 보여줍니다."
    )

    # 5. 성능 지표의 의미
    st.header("5. 성능 지표의 의미")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (결정계수)", f"{metrics['r2']:.3f}")
    m2.metric("MAE (평균절대오차)", f"{metrics['mae']:.4f}")
    m3.metric("AUC (고위험 판별)", f"{metrics['auc']:.3f}")
    st.markdown(
        f"- **R² (결정계수, 0–1)**: 학교 특성으로 휴직비율의 변동을 설명하는 비율. "
        f"현재 {metrics['r2']:.2f} → 변동의 약 **{metrics['r2'] * 100:.0f}%**를 설명. 1에 가까울수록 좋음.\n"
        f"- **MAE (평균절대오차)**: 예측 휴직비율과 실제값의 평균 차이. "
        f"현재 {metrics['mae']:.3f} → 평균 약 **{metrics['mae'] * 100:.1f}%p** 오차.\n"
        f"- **AUC (0.5–1.0)**: 고위험(상위 25%) 학교를 가려내는 판별력. 0.5=무작위, 1.0=완벽. "
        f"현재 {metrics['auc']:.2f} → **우수한 판별력**."
    )

    # 6. 전처리 과정
    st.header("6. 전처리 과정")
    st.markdown(
        "1. 6종 CSV를 `정보공시 학교코드`(+연도)로 좌측 조인\n"
        "2. 비율 파생: 기간제·보직·여교원·휴직 비율 = 해당 인원 ÷ 전체 교원\n"
        "3. 타깃(휴직비율) 결측·비정상(>1) 행 제거\n"
        "4. 결측치는 **학교급별 중앙값**으로 대치 (이상치에 둔감한 견고한 통계량)\n"
        "5. 범주형(학교급·설립구분)은 정수 인코딩\n"
        "6. 학습/평가 8:2 분할 후 LightGBM 학습 (`@st.cache_data`·`@st.cache_resource`로 캐싱)"
    )


# =============================================================================
# 라우팅 (사이드바: 관리자 대시보드 / 교사 셀프체크 / 모델·데이터 정보)
# =============================================================================
def main():
    st.set_page_config(page_title="쉼표(Swimpyo)", page_icon="🍃", layout="wide")

    # 데이터·모델 준비 (캐싱)
    df = load_features()
    model, ref_pred, explainer, median_by_level, metrics = train_model(df)
    df_t = add_tbri_columns(df, model, ref_pred)

    with st.sidebar:
        st.title("🍃 쉼표 (Swimpyo)")
        st.caption("AI 기반 교사 업무·감정 부담 조기경보 시스템")
        # 아이콘 메뉴 (라디오 대신) — 선택 항목은 검정 강조 (테마 일관성)
        mode = option_menu(
            menu_title=None,
            options=["관리자 대시보드", "교사 셀프체크", "모델·데이터 정보"],
            icons=["speedometer2", "clipboard-check", "info-circle"],  # Bootstrap Icons
            default_index=0,
            styles={
                "container": {"padding": "4px 0", "background-color": "transparent"},
                # 아이콘 색을 글자색에 맞춰 따라가게 함(inherit)
                # → 선택 시(어두운 배경) 흰색, 비선택 시 어두운색 → 둘 다 보임
                "icon": {"color": "inherit", "font-size": "16px"},
                "nav-link": {
                    "font-size": "15px", "text-align": "left", "margin": "3px 0",
                    "padding": "10px 12px", "border-radius": "8px", "color": "#262730",
                    "--hover-color": "#eceef1",
                },
                "nav-link-selected": {"background-color": "#262730", "color": "white"},
            },
        )
        st.divider()
        st.caption("데이터: 경기도교육청 학교알리미 공시(2022~2025)")
        st.caption("공개 공시데이터 — 실제 학교명·지역명 사용")

    if mode == "관리자 대시보드":
        st.title("관리자 대시보드")
        tab1, tab2, tab3 = st.tabs(["전국 위험 지도", "학교별 리포트", "What-if 시뮬레이터"])
        with tab1:
            page_admin_map(df_t)
        with tab2:
            page_admin_report(df_t, explainer)
        with tab3:
            page_admin_whatif(df_t, model, ref_pred, explainer)
    elif mode == "교사 셀프체크":
        st.title("교사 셀프체크")
        page_teacher_selfcheck(df, model, ref_pred, explainer, median_by_level)
    else:
        page_model_info(df_t, metrics)


if __name__ == "__main__":
    main()
