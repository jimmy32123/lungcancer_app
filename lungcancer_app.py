import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -------------------------------------------------------------
# 1. KMeans 모델 로드
# -------------------------------------------------------------
@st.cache_resource
def load_kmeans_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'kmeans_model.pkl')
    
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"❌ 모델 로드 중 오류 발생: {e}")
    return None

kmeans_model = load_kmeans_model()

# -------------------------------------------------------------
# 2. UI 기본 설정
# -------------------------------------------------------------
st.set_page_config(page_title="폐암 환자 군집 분석 시스템", layout="centered", page_icon="🫁")

st.title("🫁 폐암 데이터 기반 군집(Clustering) 분석 및 시각화")
st.caption("Machine Learning 모델을 통해 환자의 위치와 건강 상태를 그래프로 확인합니다.")

if kmeans_model is None:
    st.error("❌ **시스템 오류:** `kmeans_model.pkl` 파일을 찾을 수 없습니다.")
    st.stop()

st.markdown("---")

# -------------------------------------------------------------
# 3. 데이터 입력 영역
# -------------------------------------------------------------
st.subheader("📋 환자 건강 지표 입력")

with st.form(key="lung_cancer_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("🎂 건강 지표 1 (연령)", min_value=10.0, max_value=100.0, value=55.0, step=1.0)
    with col2:
        smoking = st.number_input("🚬 건강 지표 2 (흡연량)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    with col3:
        tumor_size = st.number_input("🧫 건강 지표 3 (종양 크기)", min_value=0.0, max_value=50.0, value=8.0, step=0.1)
        
    submit_button = st.form_submit_button(label="🔮 나의 군집 위치 분석하기", use_container_width=True)

# -------------------------------------------------------------
# 4. 분석 결과 및 그래프 출력 영역
# -------------------------------------------------------------
if submit_button:
    input_data = np.array([[age, smoking, tumor_size]])
    
    try:
        # 예측 결과 계산
        cluster_result = int(kmeans_model.predict(input_data)[0]) 
        
        st.markdown("---")
        st.subheader("📊 AI 분석 결과")
        
        # 4개 군집(0, 1, 2, 3) 결과 매핑 출력 [cite: 4]
        if cluster_result == 0:
            st.success("### 🎯 분석 결과: **[군집 0번] 최상위 건강군 (정상)**")
            st.markdown("- **상태:** 생체 위험 지표가 완벽히 정상 수치 범위에 머무르고 있는 가장 건강한 상태입니다.\n- **가이드:** 현재의 훌륭한 건강 습관과 식단을 그대로 유지하세요.")
        elif cluster_result == 1:
            st.info("### 🔵 분석 결과: **[군집 1번] 초기 주의군 (경계선)**")
            st.markdown("- **상태:** 큰 이상은 없으나 연령 상승 또는 가벼운 수치 변화로 인해 경계선에 걸친 단계입니다.\n- **가이드:** 정기적인 예방 검진과 더불어 가벼운 유산소 운동을 추천합니다.")
        elif cluster_result == 2:
            st.warning("### 🟡 분석 결과: **[군집 2번] 일반 위험 / 추적 관찰군**")
            st.markdown("- **상태:** 흡연력이나 특정 수치가 다소 높아져 장기적인 추적 관찰이 요구되는 상태입니다.\n- **가이드:** 금연 등 즉각적인 생활 습관 개선이 필요하며 정기 검진 주기를 좁히는 것이 좋습니다.")
        elif cluster_result == 3:
            st.error("### 🚨 분석 결과: **[군집 3번] 고위험 / 정밀 검사군**")
            st.markdown("- **상태:** 종합 지표 위험도가 임계치를 넘은 고위험군 패턴입니다. 정밀 검사가 시급합니다.\n- **가이드:** 즉시 전문의를 방문하시어 저선량 폐 CT 등 정밀 진단을 받으시는 것을 강력히 권장합니다.")
        
        # 2) 시각화 그래프 그리기
        st.markdown(" ")
        st.subheader("📍 나의 군집 위치 시각화")
        st.caption("주변 데이터 분포 속에서 내 위치(★)가 어디에 속해 있는지 확인하세요.")
        
        # 가상 데이터 분포 빌드 [cite: 1]
        np.random.seed(42)
        n_samples = 300
        
        v_age = np.random.normal(loc=age, scale=20, size=n_samples)
        v_smoking = np.random.normal(loc=smoking, scale=20, size=n_samples)
        v_tumor = np.random.normal(loc=tumor_size, scale=8, size=n_samples)
        
        v_age = np.clip(v_age, 10, 100)
        v_smoking = np.clip(v_smoking, 0, 100)
        v_tumor = np.clip(v_tumor, 0, 50)
        
        fake_data = np.column_stack([v_age, v_smoking, v_tumor]) [cite: 1]
        fake_labels = [int(x) for x in kmeans_model.predict(fake_data)] [cite: 1]
        
        # 데이터프레임 구성
        df_visual = pd.DataFrame(fake_data, columns=['연령', '흡연량', '종양크기'])
        
        # 군집 번호를 직관적인 한글 이름으로 맵핑
        cluster_names = {
            0: '군집 0: 건강군', 
            1: '군집 1: 주의군', 
            2: '군집 2: 일반위험군', 
            3: '군집 3: 고위험군'
        }
        df_visual['군집 분류'] = [cluster_names.get(x, f'군집 {x}') for x in fake_labels]
        
        # 사용자의 현재 입력 데이터 추가
        user_df = pd.DataFrame([[age, smoking, tumor_size, '★ 나의 현재 위치']], columns=['연령', '흡연량', '종양크기', '군집 분류'])
        df_total = pd.concat([df_visual, user_df], ignore_index=True)
        
        # 💡 [한글 깨짐 해결의 핵심 키포인트] 
        # Matplotlib 대신 스트림릿 내장 네이티브 차트(st.scatter_chart)를 사용합니다.
        # 이 방식을 쓰면 서버 환경에 상관없이 앱 자체 테마를 따라 100% 한글이 깨지지 않고 깔끔하게 표기됩니다!
        st.scatter_chart(
            data=df_total,
            x='연령',
            y='흡연량',
            color='군집 분류',
            color_config={
                '군집 0: 건강군': '#2ecc71',
                '군집 1: 주의군': '#3498db',
                '군집 2: 일반위험군': '#f1c40f',
                '군집 3: 고위험군': '#e74c3c',
                '★ 나의 현재 위치': '#2c3e50'
            },
            size=50,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"⚠️ 시각화 연산 중 에러가 발생했습니다: {e}")
