# Implementation Plan: 쉼표(Swimpyo)

> ACAD Stage 5 산출물 — Walking Skeleton + 단계별 구현 계획
> 입력: arch-story.md, architecture-drivers.md, architecture.md, validation-report.md
> 산출 일자: 2026-05-21
> 상태: 확정 (Claude Code 구현 단계로 인계 가능)

---

## 1. Walking Skeleton

> 아키텍처의 전체 경로를 가장 얇게 한 번 관통하는 최소 구현.
> 개발팀 내부 아키텍처 검증용. 사용자에게 출시하지 않는다.

### 대상 시나리오

**교사 셀프체크 End-to-End 최소 흐름**

> 익명 교사가 단일 항목(학교 유형)만 입력하면, 더미 LightGBM 모델이 TBRI 점수와 SHAP 값을 반환하고, LLM 게이트웨이가 (실제 LLM 호출 없이) Static Fallback 메시지를 표시한다.

이 시나리오를 선택한 이유: **단일 흐름에 아키텍처의 모든 결정이 압축됨**
- UI(Streamlit) → Service(LLM Gateway) → Domain(Model) → 파일 산출물 로드
- 화이트리스트 비식별화 (단일 필드 통과)
- LLM Static Fallback (실제 호출 없이도 작동 확인)
- `@st.cache_data` 모델 로딩 캐시
- DB 미사용 (PII 0건 구조)

### 관통하는 아키텍처 경로

```
브라우저 (HTTPS)
   ↓
Streamlit 페이지 (teacher/selfcheck_skeleton.py)
   ↓ 동기 호출
model/predict.py (인메모리 LightGBM 추론 + SHAP)
   ↓ 파일 로드 (@st.cache_data)
data/processed/model.pkl (더미 학습 1회)
   ↓
llm_gateway/call_llm.py (mock 모드: 항상 fallback)
   ↓
common/anonymizer.py (검증 통과 확인만)
   ↓
common/fallback.py (Static 응답)
   ↓
화면 결과 표시 (TBRI + SHAP + Fallback 메시지)
```

### 구현 범위 (최소)

| 계층 | Walking Skeleton 최소 구현 |
|------|---------------------|
| 프로젝트 구조 | `app.py`, `requirements.txt`, 패키지 7개 (`admin/`, `teacher/`, `chatbot/`, `llm_gateway/`, `model/`, `support_matcher/`, `common/`) — 빈 `__init__.py`만 있어도 OK |
| UI | `teacher/selfcheck_skeleton.py` — 학교 유형 입력 1개 + 제출 버튼 + 결과 영역 |
| 모델 | `model/predict.py` — `@st.cache_data` 적용된 모델 로더 + `predict(features)` 함수 |
| 산출물 | `data/processed/model.pkl` — 임의 데이터 5건으로 학습한 LightGBM (Walking Skeleton 전용) |
| LLM 게이트 | `llm_gateway/call_llm.py` — `MOCK_MODE=True` 환경 변수일 때 항상 Fallback 반환 |
| 비식별화 | `common/anonymizer.py` — 화이트리스트 1개 필드만 |
| 폴백 | `common/fallback.py` — Static 메시지 1개 |
| 테스트 | `tests/test_walking_skeleton.py` — 위 흐름 1개 통합 테스트 |
| 배포 | Streamlit Cloud 배포 1회 — URL 확인 |

### 완료 기준

다음 4개가 모두 성립하면 Walking Skeleton 완료:

1. **로컬 실행**: `streamlit run app.py` 후 브라우저에서 셀프체크 페이지 진입 → 입력 1개 제출 → 결과 표시 (TBRI 숫자 + SHAP 값 1개 + Fallback 메시지)
2. **배포 확인**: Streamlit Cloud에 푸시 후 동일 동작 확인 (HTTPS URL로 접근)
3. **통합 테스트 통과**: `pytest tests/test_walking_skeleton.py` 통과
4. **아키텍처 검증**: 모듈 간 import 방향이 ADR-001과 일치 (역방향 import가 lint 에러로 잡힘)

**예상 소요**: 1~2일 (2명 팀)

---

## 2. 의존성 기반 구현 순서

### Phase 1 (MVP) — 사용자에게 출시 가능한 최소 범위 (시연 가능 수준)

목표: 대회 시연 + 출품 PDF + Streamlit Cloud 라이브 데모

#### 1-1. 기반 인프라 (Walking Skeleton 직후, 1~2일)

| 단계 | 구현 항목 | 의존성 | 완료 기준 |
|------|---------|------|---------|
| 1-1-1 | 프로젝트 구조 확정 (패키지·모듈 골격) | Walking Skeleton | 7개 패키지 + `__init__.py` |
| 1-1-2 | `common/anonymizer.py` 화이트리스트 + 정규식 2중 방어 | 1-1-1 | 우회 시도 5종 단위 테스트 통과 |
| 1-1-3 | `common/fallback.py` Static + 룰 기반 템플릿 | 1-1-1 | 위험 요인 코드별 차등 메시지 |
| 1-1-4 | `common/log_masker.py` PII 마스킹 로깅 필터 | 1-1-1 | 학교명·전화번호 패턴 마스킹 단위 테스트 |
| 1-1-5 | `common/schema.py` 산출물 스키마 버전 검증 | 1-1-1 | 스키마 미스매치 시 명확한 에러 (R-005 완화) |
| 1-1-6 | GitHub Actions CI (pytest + flake8 + 의존성 방향 lint) | 1-1-1 | 모든 PR에서 자동 실행 |
| 1-1-7 | Streamlit secrets 설정 (OpenAI API 키, MOCK_MODE 토글) | 1-1-1 | 로컬·Cloud 환경 변수 분리 |

#### 1-2. 데이터·모델 (3~4일)

| 단계 | 구현 항목 | 의존성 | 완료 기준 |
|------|---------|------|---------|
| 1-2-1 | `etl/load_publicdata.py` — 공공데이터 CSV/Excel 로더 (실제 파일 도착 후 구체화) | 1-1-* | 학교 코드 기준 통합 데이터프레임 출력 |
| 1-2-2 | `etl/preprocess.py` — 결측치 처리·피처 엔지니어링·이상치 격리 | 1-2-1 | `features.parquet` 산출, AD-S007 검증 통과 |
| 1-2-3 | `etl/train_model.py` — LightGBM 학습 + 시계열 분할 검증 | 1-2-2 | `model.pkl` 산출, AUC ≥ 0.75 (가정 A5) |
| 1-2-4 | `etl/precompute_shap.py` — 전국 학교에 대한 SHAP 값 사전 계산 | 1-2-3 | `shap_precomputed.parquet` 산출 |
| 1-2-5 | `model/predict.py` — `@st.cache_data` 적용 인메모리 추론 | 1-2-3 | 단일 추론 200ms 이내 (AD-P004) |
| 1-2-6 | `model/explain.py` — 사전계산 SHAP 로드 + Top-N 위험 요인 | 1-2-4 | 학교 코드 → Top 5 요인 |
| 1-2-7 | `support_matcher/rules.py` — 위험 요인 코드 → 지원 프로그램 룰 | 1-2-6 | YAML로 룰 정의, 단위 테스트 |

#### 1-3. LLM 게이트웨이 (1~2일)

| 단계 | 구현 항목 | 의존성 | 완료 기준 |
|------|---------|------|---------|
| 1-3-1 | `llm_gateway/call_llm.py` 실제 OpenAI 호출 + timeout=5s | 1-1-2, 1-1-3 | AD-A003 충족 — 타임아웃 시 폴백 |
| 1-3-2 | `llm_gateway/prompts/` 시스템 프롬프트·안내 템플릿 | 1-3-1 | 챗봇 범위 명시(AD-S005·U003) |
| 1-3-3 | `llm_gateway/output_validator.py` 출력 검증 (PII·URL·인젝션) | 1-3-1 | AD-S008 — 패턴 차단 단위 테스트 |
| 1-3-4 | 적대적 케이스 20개 단위 테스트 | 1-3-2 | AD-S005 회피율 100% |

#### 1-4. 사용자 경험 — 교사 셀프체크 (2일)

| 단계 | 구현 항목 | 의존성 | 완료 기준 |
|------|---------|------|---------|
| 1-4-1 | `teacher/selfcheck.py` 5개 항목 입력 폼 | 1-2-5, 1-3-1 | AD-U001 — 4단계 이하, 30초 이내 |
| 1-4-2 | `teacher/result.py` TBRI + SHAP 차트 (Plotly) | 1-2-6 | 위험 요인 Top 5 시각화 |
| 1-4-3 | `teacher/support.py` 매칭된 지원 프로그램 표시 | 1-2-7 | 위험 요인 → 프로그램 매핑 |
| 1-4-4 | `teacher/llm_message.py` LLM 안내 메시지 + Fallback | 1-3-* | LLM 장애 시 룰 템플릿 작동 |

#### 1-5. 사용자 경험 — 관리자 대시보드 (3일)

| 단계 | 구현 항목 | 의존성 | 완료 기준 |
|------|---------|------|---------|
| 1-5-1 | `admin/auth.py` 단순 ID/PW 인증 (Streamlit secrets) | 1-1-7 | AD-S004 |
| 1-5-2 | `admin/risk_map.py` 전국 위험 지도 (Folium choropleth) | 1-2-3, 1-2-4 | A1 — 시도·시군구 히트맵, 클릭 → 상세 |
| 1-5-3 | `admin/school_report.py` 학교별 TBRI 상세 리포트 | 1-2-5, 1-2-6 | A2 — SHAP 차트 + 백분위 |
| 1-5-4 | `admin/what_if.py` What-if 시뮬레이터 | 1-2-5 | A3 — 1초 이내 반영 (AD-F004·P002) |
| 1-5-5 | `admin/support_match.py` 지원 자원 매칭 패널 | 1-2-7 | A4 |
| 1-5-6 | `admin/report_pdf.py` PDF 보고서 생성 | 1-5-2~5 | A5 — 시도교육청별 요약 |

#### 1-6. 챗봇 (1~2일)

| 단계 | 구현 항목 | 의존성 | 완료 기준 |
|------|---------|------|---------|
| 1-6-1 | `chatbot/component.py` Streamlit 챗봇 컴포넌트 (멀티턴, 세션 휘발) | 1-3-* | admin·teacher 양쪽 임베드 |
| 1-6-2 | 챗봇 통합 테스트 — 정상 케이스 + 적대적 케이스 + 범위 외 케이스 | 1-6-1, 1-3-4 | AD-U003 90% / S005 100% |

#### 1-7. 검증·시연 준비 (2일)

| 단계 | 구현 항목 | 의존성 | 완료 기준 |
|------|---------|------|---------|
| 1-7-1 | **부하 테스트** locust 50명 동시 시뮬레이션 | 1-5-*, 1-6-* | 가정 A4 검증 (R-002 완화) |
| 1-7-2 | **메모리 측정** `psutil`로 모델·데이터 로드 시 사용량 기록 | 1-2-5 | R-001 완화 데이터 확보 |
| 1-7-3 | 시연 직전 워밍업 스크립트 (모든 페이지 1회 호출) | 1-7-1 | Streamlit 캐시 사전 채움 |
| 1-7-4 | 시연 시나리오 리허설 | 1-5-*, 1-6-* | 2시간 무중단 시뮬레이션 |

#### Phase 1 완료 기준

다음 4개가 모두 성립하면 Phase 1(MVP) 완료:

1. **시연 가능**: Streamlit Cloud URL에서 관리자·교사 두 영역 + 챗봇이 정상 동작
2. **부하 검증**: 50명 동시 접속에서 다운타임 0초 (가정 A4 통과 또는 백업 호스팅 준비됨)
3. **테스트 통과**: 단위 테스트 + 통합 테스트 + 적대적 케이스 모두 통과
4. **PDF 출품**: 대회 제출용 제안서 + 발표 자료 PDF 완성

**예상 총 소요**: 약 12~15일 (2명 팀, Walking Skeleton 포함). 2주 MVP 일정에 부합.

---

### Phase 2 — 추가 기능 (대회 후)

| 단계 | 구현 항목 | 전제 조건 | 완료 기준 |
|------|---------|---------|---------|
| 2-1 | 익명 상담 예약 (AD-F008, S006) | 가정 A3 검증 통과 | PII 0건 수집 — 외부 폼 또는 일회성 토큰 |
| 2-2 | 챗봇 RAG 도입 | Phase 1 챗봇 사용량 데이터 | FAQ·매뉴얼·지원 프로그램 DB 검색 기반 응답 |
| 2-3 | 시도교육청별 멀티테넌트 | AD-C001 트래픽 임계치 도달 | 환경 변수·설정 분리만으로 인스턴스 분리 |
| 2-4 | NEIS·교원능력개발평가 연동 | 데이터 거버넌스 협의 완료 | 정식 API 인증 + 데이터 사용 동의 |
| 2-5 | 시계열 예측 (향후 1년 위험도 변화) | Phase 1 모델 성능 안정 | 시계열 모델(Prophet/시계열 LightGBM) 추가 |
| 2-6 | 모바일 PWA | UX 피드백 확보 후 | 반응형 디자인 + 오프라인 캐시 |

### Future (장기 로드맵)

- 실제 상담사 매칭 워크플로 연계
- 인과 추론 모델 (현재는 조건부 예측만 주장)
- 모바일 네이티브 앱
- 학교 단위가 아닌 교사 개인 단위 분석 (개인정보 거버넌스 별도 설계 필요)

---

## 3. Claude Code 인계 사항

다음 5개 파일을 Claude Code 프로젝트에 포함하면 Stage 5 코드 생성으로 진행할 수 있다:

1. `docs/acad/arch-story.md` — 시스템 정의 + 범위 + 시나리오 + 제약·가정
2. `docs/acad/architecture-drivers.md` — 8개 기능 드라이버 + 23개 품질 ASR + 8개 제약
3. `docs/acad/architecture.md` — 모듈러 모놀리스 + Module·C&C·Allocation 뷰
4. `docs/acad/validation-report.md` — Top 5 시나리오 + 7개 ADR + 리스크 등록부 + 기술 스택
5. `docs/acad/implementation-plan.md` — Walking Skeleton + Phase 1·2 단계별 계획 (본 문서)

**Walking Skeleton 코드 생성 명령 (참고)**:
> "위 5개 파일을 입력으로, 본 문서의 Walking Skeleton 범위(섹션 1) 코드를 생성하세요. 7개 패키지 골격 + `teacher/selfcheck_skeleton.py` + `model/predict.py` + `llm_gateway/call_llm.py` (MOCK 모드) + `common/anonymizer.py`, `common/fallback.py` + `tests/test_walking_skeleton.py`. requirements.txt와 README.md 갱신 포함."

**MVP 코드 생성 순서 (참고)**:
1-1 → 1-2 → 1-3 → 1-4 → 1-5 → 1-6 → 1-7 순서로 진행. 각 단계 완료 시 단위·통합 테스트가 통과해야 다음 단계 진입.

---

## 4. 잔존 가정사항 검증 일정

Stage 1에서 식별된 가정사항 중 미검증 항목:

| ID | 가정 | 검증 시점 | 실패 시 처리 |
|----|------|---------|----------|
| A1 | 학교 코드 기반 데이터 정상 조인 | 단계 1-2-1 직후 | 조인 키 재설계, 범위 축소 |
| A2 | 외부 LLM 비식별 프롬프트로 자연스러운 안내 생성 | 단계 1-3-2 직후 | 룰 템플릿 비중 ↑ |
| A3 | 익명 상담 예약 PII 미수집 구현 가능 | Phase 2 진입 전 | Out-of-Scope 확정 |
| A4 | Streamlit Cloud 무료 티어 50명 동시 처리 | 단계 1-7-1 (부하 테스트) | Hugging Face Spaces 등 백업 |
| A5 | 모델 AUC 0.75 이상 | 단계 1-2-3 직후 | 피처 엔지니어링 보강 또는 알고리즘 변경 |
| A6 | 챗봇 범위 통제 가능 | 단계 1-3-4 (적대적 테스트) | 시스템 프롬프트 강화 또는 챗봇 제거 |
| A7 | LLM 비용·쿼터 감당 | 시연 1주일 전 | 백업 API 키 또는 폴백 비중 ↑ |

---

## 5. ACAD 설계 영역 종료

본 문서로 ACAD Stage 1~5(설계 영역) 완료.

산출물 5종:
- ✅ `docs/acad/arch-story.md`
- ✅ `docs/acad/architecture-drivers.md`
- ✅ `docs/acad/architecture.md`
- ✅ `docs/acad/validation-report.md`
- ✅ `docs/acad/implementation-plan.md`

다음 단계: 데이터 도착 후 Claude Code에서 Walking Skeleton → Phase 1 MVP 순서로 구현 진행.
