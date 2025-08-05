import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import random
import os

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
    data = pd.read_csv('2025_급식데이터_YOLO용_면적단위 (2).csv', encoding='utf-8')
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

# Streamlit UI
st.title("🍽️ 급식 잔반량 예측기")

st.write("### 입력 정보를 바탕으로 잔반량(cm²)을 예측합니다")

제공량 = st.number_input('제공량 (kg)', min_value=0.0, max_value=20.0, value=6.0, step=0.1)
선호도 = st.slider('음식 선호도 (1~5)', 1, 5, 4)
기온 = st.number_input('기온 (°C)', min_value=-10.0, max_value=40.0, value=15.0, step=0.1)
계절 = st.selectbox('계절', ['봄', '여름', '가을', '겨울'])

st.write("### 학년별 인원 수")
남1 = st.number_input('1학년 남학생 수', min_value=0, value=30)
여1 = st.number_input('1학년 여학생 수', min_value=0, value=30)
남2 = st.number_input('2학년 남학생 수', min_value=0, value=30)
여2 = st.number_input('2학년 여학생 수', min_value=0, value=30)
남3 = st.number_input('3학년 남학생 수', min_value=0, value=30)
여3 = st.number_input('3학년 여학생 수', min_value=0, value=30)

st.write("### 행사 여부")
체육대회 = st.checkbox('체육대회')
현장체험학습 = st.checkbox('현장체험학습')
점심행사 = st.checkbox('점심행사')

if st.button("예측하기"):
    # 입력 데이터 구성
    new_input = pd.DataFrame([{
        '제공량(kg)': 제공량,
        '선호도': 선호도,
        '기온(°C)': 기온,
        '계절': 계절,
        '1학년_남': 남1,
        '1학년_여': 여1,
        '2학년_남': 남2,
        '2학년_여': 여2,
        '3학년_남': 남3,
        '3학년_여': 여3,
        '체육대회': int(체육대회),
        '현장체험학습': int(현장체험학습),
        '점심행사': int(점심행사)
    }])

    # 전처리
    new_encoded = pd.get_dummies(new_input, columns=['계절', '점심행사'])
    new_encoded = new_encoded.reindex(columns=st.session_state.X_encoded.columns, fill_value=0)
    new_encoded[st.session_state.num_cols] = st.session_state.scaler.transform(new_encoded[st.session_state.num_cols])

    # 예측
    predicted = st.session_state.model.predict(new_encoded)[0][0]
    st.success(f"📊 예측된 잔반량: {predicted:.2f} cm²")
