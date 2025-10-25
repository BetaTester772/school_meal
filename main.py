import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib
from matplotlib import font_manager, rc
import random
import os

# 폰트 설정
# font_path = '/usr/share/fonts/nanumfont/NanumGothic.ttf'
# fontprop = font_manager.FontProperties(fname=font_path)
# rc('font', family=fontprop.get_name())
# matplotlib.rcParams['axes.unicode_minus'] = False

# 초기 실행 설정 (세션 상태를 통해 최초 1회만 실행)
if 'initialized' not in st.session_state:
    print("hello")


    # 시드 고정 함수
    def set_seed(seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = "1"
        os.environ['TF_CUDNN_DETERMINISM'] = "1"
        os.environ['PYTHONHASHSEED'] = str(seed)


    set_seed()

    # 모델 로드
    st.session_state.model = load_model('leftover_prediction_model.keras')

    # 데이터셋 로드 및 전처리 기준 확보
    data = pd.read_csv('predict_data.csv', encoding='utf-8')
    data.columns = data.columns.str.strip()

    X = data[['선호도', '기온(°C)', '계절',
              '1학년_남', '1학년_여',
              '2학년_남', '2학년_여',
              '3학년_남', '3학년_여',
              '체육대회', '현장체험학습', '점심행사',
              '제공량(kg)']]

    X_encoded = pd.get_dummies(X, columns=['계절', '점심행사'])

    scaler = StandardScaler()
    num_cols = ['선호도', '기온(°C)', '1학년_남', '1학년_여', '2학년_남', '2학년_여', '3학년_남', '3학년_여', '제공량(kg)']
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

    # 세션 상태 저장
    st.session_state.X_encoded = X_encoded
    st.session_state.scaler = scaler
    st.session_state.num_cols = num_cols
    st.session_state.initialized = True

# 메인 타이틀
st.title("급식 AI 통합 분석 시스템")

# 탭 생성
tab1, tab2 = st.tabs(["잔반율 예측기", "메뉴 분석 대시보드"])

# ==================== 탭 1: 잔반율 예측기 ====================
with tab1:
    st.header("급식 잔반율 예측기")
    st.write("### 메뉴를 선택하고 환경 변수를 입력하면 잔반량을 예측합니다")

    # predict_data.csv 로드
    predict_df = pd.read_csv('predict_data.csv', encoding='utf-8')
    predict_df.columns = predict_df.columns.str.strip()

    # recommendation_data.csv 로드 (대체 메뉴 추천용)
    recommend_df = pd.read_csv('recommendation_data.csv', encoding='utf-8')
    recommend_df.columns = recommend_df.columns.str.strip()

    # 메뉴 선택
    메뉴_목록 = predict_df['메뉴'].unique()
    선택_메뉴 = st.selectbox('메뉴를 선택하세요', 메뉴_목록, key='menu_select')

    # 선택된 메뉴의 정보 추출
    selected_menu_data = predict_df[predict_df['메뉴'] == 선택_메뉴].iloc[0]
    메뉴_선호도 = selected_menu_data['선호도']
    메뉴_제공량 = selected_menu_data['제공량(kg)']

    st.info(f"**선택된 메뉴:** {선택_메뉴} (기본 선호도: {메뉴_선호도}, 기본 제공량: {메뉴_제공량:.2f}kg)")

    # 환경 변수 입력
    st.write("### 환경 변수 입력")

    col_env1, col_env2 = st.columns(2)
    with col_env1:
        기온 = st.number_input('기온 (°C)', min_value=-10.0, max_value=40.0, value=15.0, step=0.1)
        계절 = st.selectbox('계절', ['봄', '여름', '가을', '겨울'])
    with col_env2:
        점심행사 = st.checkbox('점심행사')
        체육대회 = st.checkbox('체육대회')
        현장체험학습 = st.checkbox('현장체험학습')

    st.write("### 학년별 인원 수")
    col1, col2 = st.columns(2)
    with col1:
        남1 = st.number_input('1학년 남학생 수', min_value=0, value=30)
        남2 = st.number_input('2학년 남학생 수', min_value=0, value=30)
        남3 = st.number_input('3학년 남학생 수', min_value=0, value=30)
    with col2:
        여1 = st.number_input('1학년 여학생 수', min_value=0, value=30)
        여2 = st.number_input('2학년 여학생 수', min_value=0, value=30)
        여3 = st.number_input('3학년 여학생 수', min_value=0, value=30)

    # 임계값 설정
    st.write("### 잔반량 관리")
    임계값 = st.slider("잔반량 임계값 (cm²)", min_value=0, max_value=3000, value=1500, step=100)

    if st.button("예측하기", type="primary"):
        # 입력 데이터 구성
        new_input = pd.DataFrame([{
                '제공량(kg)': 메뉴_제공량,
                '선호도'    : 메뉴_선호도,
                '기온(°C)' : 기온,
                '계절'     : 계절,
                '1학년_남'  : 남1,
                '1학년_여'  : 여1,
                '2학년_남'  : 남2,
                '2학년_여'  : 여2,
                '3학년_남'  : 남3,
                '3학년_여'  : 여3,
                '체육대회'   : int(체육대회),
                '현장체험학습' : int(현장체험학습),
                '점심행사'   : int(점심행사)
        }])

        # 전처리
        new_encoded = pd.get_dummies(new_input, columns=['계절', '점심행사'])
        new_encoded = new_encoded.reindex(columns=st.session_state.X_encoded.columns, fill_value=0)
        new_encoded[st.session_state.num_cols] = st.session_state.scaler.transform(
                new_encoded[st.session_state.num_cols])

        # 예측
        predicted = st.session_state.model.predict(new_encoded)[0][0]

        # 잔반율 계산: (predicted / (제공량 * 100)) * 100
        잔반율 = (predicted / (메뉴_제공량 * 100)) * 100

        # 예측 결과 표시
        st.divider()
        st.subheader(f"예측 결과: {선택_메뉴}")

        col_result1, col_result2 = st.columns(2)
        with col_result1:
            st.metric("예측된 잔반량", f"{predicted:.2f} cm²")
        with col_result2:
            st.metric("예측된 잔반율", f"{잔반율:.2f}%")

        # 임계값 비교 및 대체 메뉴 추천
        if predicted >= 임계값:
            st.warning(f"⚠️ 예측된 잔반량({predicted:.2f} cm²)이 임계값({임계값} cm²)을 초과합니다!")

            st.write("### 🔄 대체 메뉴 추천")

            # recommendation_data.csv에서 현재 메뉴 찾기
            if 선택_메뉴 in recommend_df['식품명'].values:
                nutrient_cols = [
                    "에너지(kcal)", "단백질(g)", "지방(g)",
                    "탄수화물(g)", "당류(g)", "식이섬유(g)",
                    "칼슘(mg)", "철(mg)", "인(mg)", "칼륨(mg)",
                    "나트륨(mg)", "비타민 A(μg RAE)", "비타민 C(mg)", "비타민 D(μg)"
                ]

                # 영양소 데이터 전처리
                nutrients = recommend_df[nutrient_cols].copy()
                nutrients = nutrients.apply(pd.to_numeric, errors='coerce').fillna(nutrients.mean())

                # 선택된 메뉴의 인덱스
                menu_idx = recommend_df[recommend_df['식품명'] == 선택_메뉴].index[0]

                # 코사인 유사도 계산
                similarity_matrix = cosine_similarity(nutrients)
                similar_idx = similarity_matrix[menu_idx].argsort()[::-1][1:]  # 자신 제외

                # 잔반량이 낮으면서 유사도가 높은 메뉴 추천
                candidates = recommend_df.loc[similar_idx]
                candidates = candidates[candidates["평균잔반량"] < 임계값].head(5)

                if not candidates.empty:
                    st.success("다음의 대체 메뉴를 추천합니다:")
                    for idx, (_, cand) in enumerate(candidates.iterrows(), 1):
                        col_cand1, col_cand2, col_cand3 = st.columns([2, 1, 1])
                        with col_cand1:
                            st.write(f"**{idx}. {cand['식품명']}**")
                        with col_cand2:
                            st.write(f"평균잔반량: {cand['평균잔반량']:.1f} cm²")
                        with col_cand3:
                            st.write(f"에너지: {cand['에너지(kcal)']:.0f} kcal")
                else:
                    st.info("적절한 대체 메뉴를 찾지 못했습니다. 임계값을 조정해보세요.")
            else:
                st.info("해당 메뉴는 추천 데이터에 없어 대체 메뉴를 제시할 수 없습니다.")
        else:
            st.success(f"✓ 잔반량이 적절합니다. (임계값: {임계값} cm²)")

# ==================== 탭 2: 메뉴 분석 대시보드 ====================
with tab2:
    st.header("AI 기반 급식 메뉴 분석 대시보드")

    # 1. 데이터 불러오기
    uploaded_file = st.file_uploader("CSV 파일(UTF-8) 업로드", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='cp949')
    else:
        df = pd.read_csv("recommendation_data.csv", encoding='utf-8')
        st.info("기본 파일을 불러왔습니다. 업로드된 파일이 없으므로 기본 데이터를 사용합니다.")

    df.columns = df.columns.str.strip()  # 공백 제거

    # 2. 영양소 선택
    nutrient_cols = [
            "에너지(kcal)", "단백질(g)", "지방(g)",
            "탄수화물(g)", "당류(g)", "식이섬유(g)",
            "칼슘(mg)", "철(mg)", "인(mg)", "칼륨(mg)",
            "나트륨(mg)", "비타민 A(μg RAE)", "비타민 C(mg)", "비타민 D(μg)"
    ]
    nutrients = df[nutrient_cols].copy()
    nutrients = nutrients.apply(pd.to_numeric, errors='coerce').fillna(nutrients.mean())

    # 3. 코사인 유사도 기반 대체 메뉴 추천
    st.header("잔반률 높은 메뉴의 대체 추천")

    threshold = st.slider("잔반률 기준값 (이상)", min_value=0, max_value=200, value=100)

    # 잔반률 높은 메뉴 추출
    low_pref_menus = df[df["평균잔반량"] >= threshold].reset_index()

    if not low_pref_menus.empty:
        # 선택 박스
        selected_menu = st.selectbox("대체 추천 받을 메뉴를 선택하세요", low_pref_menus["식품명"])

        # 선택된 메뉴의 인덱스 찾기
        menu_idx = df[df["식품명"] == selected_menu].index[0]

        # 유사도 계산
        similarity_matrix = cosine_similarity(nutrients)
        similar_idx = similarity_matrix[menu_idx].argsort()[::-1][1:]  # 자기 제외

        # 잔반률 낮은 후보 중 유사도 높은 상위 5개
        candidates = df.loc[similar_idx]
        candidates = candidates[candidates["평균잔반량"] < threshold].head(5)

        st.subheader(f"'{selected_menu}' 대신 추천할 수 있는 메뉴:")
        if candidates.empty:
            st.write(" - 적절한 대체 메뉴 없음")
        else:
            for _, cand in candidates.iterrows():
                st.write(f" - {cand['식품명']} (평균잔반량 {cand['평균잔반량']:.2f})")
    else:
        st.warning("기준 이상 잔반 메뉴가 없습니다. 슬라이더를 조정해 보세요.")

    # 4. 막대 그래프
    st.header(f"평균잔반량 {threshold} 이상 식품 그래프")

    high_leftover = df[df["평균잔반량"] >= threshold]

    if high_leftover.empty:
        st.warning("해당 기준 이상 잔반량을 가진 식품이 없습니다.")
    else:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(high_leftover["식품명"], high_leftover["평균잔반량"], color='tomato')
        ax1.set_ylabel("평균잔반량")
        ax1.set_title(f"평균잔반량 {threshold} 이상 식품")
        plt.xticks(rotation=45, ha='right', fontsize=10)
        st.pyplot(fig1)

    # 5. 군집화 분석
    # 평균잔반량 기준 이상 메뉴들 추출
    high_leftover_menus = df[df["평균잔반량"] >= threshold]["식품명"].tolist()

    # 유사도 기반 추천 메뉴 5개씩 수집
    similarity_matrix = cosine_similarity(nutrients)
    menu_indices = df[df["식품명"].isin(high_leftover_menus)].index

    recommended_menus = set()
    for idx in menu_indices:
        similar_idx = similarity_matrix[idx].argsort()[::-1][1:]  # 자기 자신 제외
        similar_candidates = df.loc[similar_idx]
        candidates = similar_candidates[similar_candidates["평균잔반량"] < threshold].head(5)
        recommended_menus.update(candidates["식품명"].tolist())

    # 군집화 대상: 원 메뉴 + 추천 메뉴
    selected_for_clustering = list(set(high_leftover_menus) | recommended_menus)
    cluster_targets = df[df["식품명"].isin(selected_for_clustering)].copy()

    st.header("평균잔반량 기준 메뉴 + 추천 메뉴 3D 군집화")

    if cluster_targets.empty:
        st.warning("클러스터링을 위한 식품이 없습니다. 기준을 낮춰보세요.")
    else:
        # 영양소 추출 및 전처리
        nutrient_data = cluster_targets[nutrient_cols].copy()
        nutrient_data = nutrient_data.apply(pd.to_numeric, errors='coerce').fillna(nutrient_data.mean())
        scaler_cluster = StandardScaler()
        nutrient_scaled = scaler_cluster.fit_transform(nutrient_data)

        # 클러스터 수 선택
        k = st.slider("클러스터 개수 (k)", min_value=2, max_value=min(10, len(cluster_targets)), value=3)

        # KMeans 클러스터링
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(nutrient_scaled)

        # PCA 3차원 축소
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(nutrient_scaled)

        cluster_targets["cluster"] = clusters
        cluster_targets["PCA1"] = pca_result[:, 0]
        cluster_targets["PCA2"] = pca_result[:, 1]
        cluster_targets["PCA3"] = pca_result[:, 2]

        # 시각화
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        scatter = ax2.scatter(
                cluster_targets["PCA1"],
                cluster_targets["PCA2"],
                cluster_targets["PCA3"],
                c=cluster_targets["cluster"],
                cmap="Set2",
                s=100,
        )

        for i in range(len(cluster_targets)):
            ax2.text(
                    cluster_targets.iloc[i]["PCA1"],
                    cluster_targets.iloc[i]["PCA2"],
                    cluster_targets.iloc[i]["PCA3"],
                    cluster_targets.iloc[i]["식품명"],
                    fontsize=7,
            )

        ax2.set_title("잔반량 높은 메뉴 + 추천 메뉴 군집화 (3D PCA)")
        ax2.set_xlabel("PCA1")
        ax2.set_ylabel("PCA2")
        ax2.set_zlabel("PCA3")

        plt.legend(*scatter.legend_elements(), title="클러스터")
        plt.tight_layout()
        st.pyplot(fig2)