import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wave Dashboard", layout="wide")

st.title("🌊 Wave Monitoring Dashboard")

# -------------------------------
# 파일 업로드
# -------------------------------
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

# -------------------------------
# 사이드바 설정
# -------------------------------
with st.sidebar:
    st.header("⚙ 필터 설정")
    max_wave = st.number_input("Wave Height 최대값", value=7.0)
    max_sig_wave = st.number_input("Significant Wave Height 최대값", value=10.0)
    max_period = st.number_input("Wave Period 최대값", value=12.0)

# -------------------------------
# 데이터 처리
# -------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # 시간 변환
    if "Time and Date" in df.columns:
        df["Time and Date"] = pd.to_datetime(df["Time and Date"], errors="coerce")
        df = df.dropna(subset=["Time and Date"])

    # 숫자 변환
    cols = ["H_wave", "Hs_wave", "T_wave", "AquaPro 400 Temperature"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------------
    # 필터 적용
    # -------------------------------
    if "H_wave" in df.columns:
        df.loc[df["H_wave"] > max_wave, "H_wave"] = np.nan

    if "Hs_wave" in df.columns:
        df.loc[df["Hs_wave"] > max_sig_wave, "Hs_wave"] = np.nan

    if "T_wave" in df.columns:
        df.loc[df["T_wave"] > max_period, "T_wave"] = np.nan

    # -------------------------------
    # 기본 정보
    # -------------------------------
    st.subheader("📊 데이터 정보")
    st.write("데이터 개수:", len(df))

    col1, col2, col3 = st.columns(3)

    if "Hs_wave" in df.columns:
        col1.metric("평균 Hs_wave", f"{df['Hs_wave'].mean():.2f}")

    if "H_wave" in df.columns:
        col2.metric("최대 H_wave", f"{df['H_wave'].max():.2f}")

    if "AquaPro 400 Temperature" in df.columns:
        col3.metric("평균 온도", f"{df['AquaPro 400 Temperature'].mean():.2f}")

    # -------------------------------
    # 그래프
    # -------------------------------
    st.subheader("📈 그래프")

    if "Time and Date" in df.columns:

        if "Hs_wave" in df.columns:
            fig1, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(df["Time and Date"], df["Hs_wave"])
            ax1.set_title("Hs_wave")
            st.pyplot(fig1)

        if "AquaPro 400 Temperature" in df.columns:
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(df["Time and Date"], df["AquaPro 400 Temperature"])
            ax2.set_title("Temperature")
            st.pyplot(fig2)

    # -------------------------------
    # 데이터 테이블
    # -------------------------------
    st.subheader("📋 데이터 미리보기")
    st.dataframe(df.head(100))

else:
    st.info("CSV 파일을 업로드하세요.")