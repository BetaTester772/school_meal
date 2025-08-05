import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib

# 한글 폰트 설정
import matplotlib.font_manager as fm

path = '/usr/share/fonts/nanumfont/NanumGothic.ttf'
fontprop = fm.FontProperties(fname=path, size=18)
matplotlib.rc('font', family=fontprop.get_name())
matplotlib.rcParams['axes.unicode_minus'] = False

st.title("🍽️ AI 기반 급식 메뉴 분석 대시보드")

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
st.header("🥄 잔반률 높은 메뉴의 대체 추천")

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

    st.subheader(f"👉 '{selected_menu}' 대신 추천할 수 있는 메뉴:")
    if candidates.empty:
        st.write(" - ❗ 적절한 대체 메뉴 없음")
    else:
        for _, cand in candidates.iterrows():
            st.write(f" - {cand['식품명']} (평균잔반량 {cand['평균잔반량']:.2f})")
else:
    st.warning("⚠️ 기준 이상 잔반 메뉴가 없습니다. 슬라이더를 조정해 보세요.")

# 4. 막대 그래프
st.header(f"📊 평균잔반량 {threshold} 이상 식품 그래프")

high_leftover = df[df["평균잔반량"] >= threshold]

if high_leftover.empty:
    st.warning("해당 기준 이상 잔반량을 가진 식품이 없습니다.")
else:
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(high_leftover["식품명"], high_leftover["평균잔반량"], color='tomato', fontproperties=fontprop)
    ax1.set_ylabel("평균잔반량", fontproperties=fontprop)
    ax1.set_title(f"평균잔반량 {threshold} 이상 식품", fontproperties=fontprop)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    st.pyplot(fig1)

# 1. 평균잔반량 기준 이상 메뉴들 추출
high_leftover_menus = df[df["평균잔반량"] >= threshold]["식품명"].tolist()

# 2. 유사도 기반 추천 메뉴 5개씩 수집
similarity_matrix = cosine_similarity(nutrients)
menu_indices = df[df["식품명"].isin(high_leftover_menus)].index

recommended_menus = set()
for idx in menu_indices:
    similar_idx = similarity_matrix[idx].argsort()[::-1][1:]  # 자기 자신 제외
    similar_candidates = df.loc[similar_idx]
    candidates = similar_candidates[similar_candidates["평균잔반량"] < threshold].head(5)
    recommended_menus.update(candidates["식품명"].tolist())

# 3. 군집화 대상: 원 메뉴 + 추천 메뉴
selected_for_clustering = list(set(high_leftover_menus) | recommended_menus)
cluster_targets = df[df["식품명"].isin(selected_for_clustering)].copy()

st.header("📌 평균잔반량 기준 메뉴 + 추천 메뉴 3D 군집화")

if cluster_targets.empty:
    st.warning("📌 클러스터링을 위한 식품이 없습니다. 기준을 낮춰보세요.")
else:
    # 영양소 추출 및 전처리
    nutrient_data = cluster_targets[nutrient_cols].copy()
    nutrient_data = nutrient_data.apply(pd.to_numeric, errors='coerce').fillna(nutrient_data.mean())
    scaler = StandardScaler()
    nutrient_scaled = scaler.fit_transform(nutrient_data)

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
            fontproperties=fontprop
    )

    for i in range(len(cluster_targets)):
        ax2.text(
                cluster_targets.iloc[i]["PCA1"],
                cluster_targets.iloc[i]["PCA2"],
                cluster_targets.iloc[i]["PCA3"],
                cluster_targets.iloc[i]["식품명"],
                fontsize=7,
                fontproperties=fontprop
        )

    ax2.set_title("잔반량 높은 메뉴 + 추천 메뉴 군집화 (3D PCA)", fontproperties=fontprop)
    ax2.set_xlabel("PCA1")
    ax2.set_ylabel("PCA2")
    ax2.set_zlabel("PCA3")

    plt.legend(*scatter.legend_elements(), title="클러스터", fontproperties=fontprop)
    plt.tight_layout()
    st.pyplot(fig2)
