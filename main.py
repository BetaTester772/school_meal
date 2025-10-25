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
font_path = '/usr/share/fonts/nanumfont/NanumGothic.ttf'
fontprop = font_manager.FontProperties(fname=font_path)
rc('font', family=fontprop.get_name())
matplotlib.rcParams['axes.unicode_minus'] = False

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
tab1, tab2 = st.tabs(["잔반율 예측", "메뉴 분석"])

# ==================== 탭 1: 잔반율 예측 ====================
with tab1:
    st.header("급식 잔반율 예측 및 대체 메뉴 추천")
    st.write("메뉴를 선택하고 환경 변수를 입력하면 잔반율을 예측하고 대체 메뉴를 추천합니다")

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

    st.divider()

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
    st.write("### 잔반율 관리")
    잔반율_임계값 = st.slider("잔반율 임계값 (%)", min_value=0, max_value=100, value=50, step=5)

    st.divider()

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
        if 잔반율 >= 잔반율_임계값:
            st.warning(f"⚠️ 예측된 잔반율({잔반율:.2f}%)이 임계값({잔반율_임계값}%)을 초과합니다!")

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

                # 잔반율이 낮으면서 유사도가 높은 메뉴 추천
                # 대체 메뉴의 예상 잔반율 계산
                candidates = recommend_df.loc[similar_idx].copy()
                candidates['예상_잔반율'] = (candidates["평균잔반량"] / (메뉴_제공량 * 100)) * 100
                candidates = candidates[candidates['예상_잔반율'] < 잔반율_임계값].head(5)

                if not candidates.empty:
                    st.success("다음의 대체 메뉴를 추천합니다:")
                    for idx, (_, cand) in enumerate(candidates.iterrows(), 1):
                        col_cand1, col_cand2, col_cand3 = st.columns([2, 1, 1])
                        with col_cand1:
                            st.write(f"**{idx}. {cand['식품명']}**")
                        with col_cand2:
                            st.write(f"예상 잔반율: {cand['예상_잔반율']:.1f}%")
                        with col_cand3:
                            st.write(f"에너지: {cand['에너지(kcal)']:.0f} kcal")
                else:
                    st.info("적절한 대체 메뉴를 찾지 못했습니다. 임계값을 조정해보세요.")
            else:
                st.info("해당 메뉴는 추천 데이터에 없어 대체 메뉴를 제시할 수 없습니다.")
        else:
            st.success(f"✓ 잔반율이 적절합니다. (임계값: {잔반율_임계값}%)")

# ==================== 탭 2: 메뉴 분석 ====================
with tab2:
    st.header("메뉴 분석 - 3D 군집화")
    st.write("음식의 영양소 정보를 기반으로 메뉴를 분류합니다")

    # 데이터 불러오기
    df = pd.read_csv("recommendation_data.csv", encoding='utf-8')
    df.columns = df.columns.str.strip()

    # 영양소 열 정의
    nutrient_cols = [
        "에너지(kcal)", "단백질(g)", "지방(g)",
        "탄수화물(g)", "당류(g)", "식이섬유(g)",
        "칼슘(mg)", "철(mg)", "인(mg)", "칼륨(mg)",
        "나트륨(mg)", "비타민 A(μg RAE)", "비타민 C(mg)", "비타민 D(μg)"
    ]

    # 영양소 데이터 전처리
    nutrient_data = df[nutrient_cols].copy()
    nutrient_data = nutrient_data.apply(pd.to_numeric, errors='coerce').fillna(nutrient_data.mean())

    # 클러스터 수 선택
    k = st.slider("클러스터 개수 (k)", min_value=2, max_value=10, value=3)

    # 스케일링
    scaler_cluster = StandardScaler()
    nutrient_scaled = scaler_cluster.fit_transform(nutrient_data)

    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(nutrient_scaled)

    # PCA 3차원 축소
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(nutrient_scaled)

    # 데이터프레임에 클러스터 정보 추가
    df["cluster"] = clusters
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]
    df["PCA3"] = pca_result[:, 2]

    # 3D 시각화
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        df["PCA1"],
        df["PCA2"],
        df["PCA3"],
        c=df["cluster"],
        cmap="Set2",
        s=100,
        alpha=0.6
    )

    # 메뉴명 라벨 추가
    for i in range(len(df)):
        ax.text(
            df.iloc[i]["PCA1"],
            df.iloc[i]["PCA2"],
            df.iloc[i]["PCA3"],
            df.iloc[i]["식품명"],
            fontsize=8
        )

    ax.set_title("메뉴 영양소 기반 3D 군집화 (PCA)")
    ax.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PCA3 ({pca.explained_variance_ratio_[2]:.1%})")

    plt.legend(*scatter.legend_elements(), title="클러스터", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

    # 클러스터별 메뉴 정보 표시
    st.divider()
    st.write("### 클러스터별 메뉴")
    for cluster_id in range(k):
        cluster_items = df[df["cluster"] == cluster_id]["식품명"].tolist()
        st.write(f"**클러스터 {cluster_id + 1}:** {', '.join(cluster_items)}")
