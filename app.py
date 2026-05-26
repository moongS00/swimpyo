# =============================================================================
# 쉼표(Swimpyo) — AI 기반 교사 업무·감정 부담 조기경보 시스템 (프로토타입)
# 제8회 교육 공공데이터 AI 활용대회 출품작
# =============================================================================
#
# [아키텍처]  모듈러 모놀리스 + 파일 기반 산출물 (docs/acad/architecture.md, ADR-001)
#   본 단일 파일은 ACAD 설계의 논리 모듈을 섹션으로 반영한다:
#     [common]  설정·식별정보 익명화·유틸
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
# [프라이버시]  데이터 로드 '직후' 학교명·지역명을 코드 내부에서 익명화 (AD-S001)
#   - 화면에는 원본 학교명/지역명을 일절 노출하지 않음 (예: A고등학교, B지역)
#   - 개인 식별 정보 미수집·미저장 (DB 미사용, 세션 휘발성)
# =============================================================================

import os
import glob
import string

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
# [common] 식별 정보 익명화
#   - 데이터 로드 직후 호출하여 학교명·지역명을 결정적(deterministic)으로 마스킹
#   - 같은 학교코드는 항상 같은 익명 라벨로 매핑 (세션 내 일관성)
# =============================================================================
def _region_alias(idx: int) -> str:
    """0,1,2,... → '지역 A','지역 B',... 'B시' 형태의 익명 지역 라벨."""
    letters = string.ascii_uppercase
    if idx < 26:
        return f"{letters[idx]}지역"
    first, second = divmod(idx, 26)
    return f"{letters[first - 1]}{letters[second]}지역"


def anonymize(df: pd.DataFrame) -> pd.DataFrame:
    """학교명·지역명을 익명 라벨로 치환한 새 컬럼을 추가하고 원본 식별 컬럼은 제거.
    화면 출력에는 익명 컬럼(학교, 지역)만 사용한다."""
    df = df.copy()

    # 지역(시군구) → 'A지역','B지역' ... 첫 등장 순서로 결정적 매핑
    region_src = df["지역"].fillna("미상")
    uniq_regions = {r: _region_alias(i) for i, r in enumerate(sorted(region_src.unique()))}
    df["지역"] = region_src.map(uniq_regions)

    # 학교명 → '익명{급}-{번호}' (학교코드 기준 결정적). 원본 학교명은 버린다.
    suffix = {0: "초", 1: "중", 2: "고"}
    codes = sorted(df[KEY].unique())
    code_rank = {c: i + 1 for i, c in enumerate(codes)}
    df["학교"] = df.apply(
        lambda r: f"{suffix.get(r['학교급코드'], '교')}-{code_rank[r[KEY]]:04d}", axis=1
    )

    # 원본 식별 컬럼 제거 (학교명, 학교코드는 화면 미노출)
    df = df.drop(columns=[c for c in ["학교명", KEY] if c in df.columns])
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

    # 식별정보 익명화 (로드 직후) — 이후 화면에는 익명 컬럼만 사용
    base = anonymize(base)
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
    st.caption("학교 위치는 공시 위경도 기반이며, 화면 라벨은 모두 익명 처리되었습니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        lvl = st.selectbox("학교급", [2, 1, 0], format_func=lambda x: LEVEL_NAME[x])
    with c2:
        years = sorted(df_t["연도"].unique())
        year = st.selectbox("연도", years, index=len(years) - 1)
    with c3:
        grades = st.multiselect("등급 필터", ["적색", "황색", "녹색"],
                                default=["적색", "황색", "녹색"])

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

    # 고위험 상위 표 (익명)
    st.markdown("##### 고위험 학교 TOP 15 (익명)")
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
        school = st.selectbox("학교(익명)", sub["학교"].tolist(), key="rep_school")

    row = sub[sub["학교"] == school].iloc[[0]]
    grade = row["등급"].iloc[0]
    tbri = row["TBRI"].iloc[0]

    left, right = st.columns([1, 1.3])
    with left:
        st.plotly_chart(plot_gauge(tbri, grade), width="stretch", key="gauge_report")
        st.metric("소속(익명 지역)", row["지역"].iloc[0])
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
        school = st.selectbox("기준 학교(익명)", sub["학교"].tolist(), key="wi_school")

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
# 라우팅 (사이드바: 관리자 대시보드 / 교사 셀프체크)
# =============================================================================
def main():
    st.set_page_config(page_title="쉼표(Swimpyo)", page_icon="🍃", layout="wide")

    # 데이터·모델 준비 (캐싱)
    df = load_features()
    model, ref_pred, explainer, median_by_level, metrics = train_model(df)
    df_t = add_tbri_columns(df, model, ref_pred)

    st.sidebar.title("🍃 쉼표 (Swimpyo)")
    st.sidebar.caption("AI 기반 교사 업무·감정 부담 조기경보 시스템")
    mode = st.sidebar.radio(
        "메뉴",
        ["관리자 대시보드", "교사 셀프체크"],
    )
    st.sidebar.divider()
    with st.sidebar.expander("ℹ️ 모델 정보"):
        st.write(f"학습 표본: {metrics['n']:,}개 (학교×연도)")
        st.write(f"휴직비율 회귀 R²: {metrics['r2']:.3f}")
        st.write(f"고위험 분류 AUC: {metrics['auc']:.3f}")
        st.write("모델: LightGBM · 설명: SHAP")
    st.sidebar.caption("데이터: 경기도교육청 학교알리미 공시(2022~2025)")
    st.sidebar.caption("화면의 학교명·지역명은 모두 익명 처리됨")

    if mode == "관리자 대시보드":
        st.title("관리자 대시보드")
        tab1, tab2, tab3 = st.tabs(["전국 위험 지도", "학교별 리포트", "What-if 시뮬레이터"])
        with tab1:
            page_admin_map(df_t)
        with tab2:
            page_admin_report(df_t, explainer)
        with tab3:
            page_admin_whatif(df_t, model, ref_pred, explainer)
    else:
        st.title("교사 셀프체크")
        page_teacher_selfcheck(df, model, ref_pred, explainer, median_by_level)


if __name__ == "__main__":
    main()
