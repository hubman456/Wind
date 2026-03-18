import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(page_title="Wave Monitoring Dashboard", layout="wide")
st.title("🌊 Wave Monitoring Dashboard")

# ============================================================
# 사이드바 필터
# ============================================================
with st.sidebar:
    st.header("⚙ 필터 설정")

    max_wave = st.number_input("Wave Height 최대값", value=7.0)
    max_sig_wave = st.number_input("Significant Wave Height 최대값", value=10.0)
    max_period = st.number_input("Wave Period 최대값", value=12.0)

# ============================================================
# 여러 파일 업로드
# ============================================================
uploaded_files = st.file_uploader(
    "CSV 파일 업로드 (여러 개 가능)",
    type=["csv"],
    accept_multiple_files=True
)

# ============================================================
# 유틸 함수
# ============================================================
def find_time_column(df):
    candidates = [
        "Time and Date", "Time", "Datetime", "DateTime", "Timestamp",
        "time", "datetime", "date_time"
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None

def get_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def circular_mean_deg(series):
    """파향 평균용 원형평균"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan

    rad = np.deg2rad(s)
    mean_sin = np.sin(rad).mean()
    mean_cos = np.cos(rad).mean()
    angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    return (angle + 360) % 360

def parse_datetime_series(series):
    """
    날짜 자동 파싱
    - YYYY.MM.DD HH:MM:SS
    - YYYY-MM-DD HH:MM:SS
    - DD/MM/YYYY HH:MM:SS
    등을 최대한 안전하게 처리
    """
    s = series.astype(str).str.strip()

    # 1차: 일반 파싱
    dt = pd.to_datetime(s, errors="coerce")

    # 2차: dayfirst=True 재시도
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(s[mask], errors="coerce", dayfirst=True)
        dt.loc[mask] = dt2

    # 3차: 자주 쓰는 형식 직접 지정
    known_formats = [
        "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]

    for fmt in known_formats:
        mask = dt.isna()
        if not mask.any():
            break
        try:
            dt_try = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            dt.loc[mask] = dt_try
        except Exception:
            pass

    return dt

# ============================================================
# 메인
# ============================================================
if uploaded_files:
    dfs = []

    for file in uploaded_files:
        try:
            df_temp = pd.read_csv(file)
            df_temp.columns = df_temp.columns.str.strip()
            df_temp["source_file"] = file.name
            dfs.append(df_temp)
        except Exception as e:
            st.warning(f"{file.name} 읽기 실패: {e}")

    if len(dfs) == 0:
        st.error("읽을 수 있는 CSV 파일이 없습니다.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)

    # ========================================================
    # 컬럼 찾기
    # ========================================================
    time_col = find_time_column(df)

    temp_col = get_existing_col(df, [
        "AquaPro 400 Temperature", "Temperature", "Temp", "temperature"
    ])
    period_col = get_existing_col(df, [
        "T_wave", "Wave Period", "Period", "Tz"
    ])
    sig_wave_col = get_existing_col(df, [
        "Hs_wave", "Hm0", "Significant Wave Height"
    ])
    wave_col = get_existing_col(df, [
        "H_wave", "Wave Height", "Hmax"
    ])
    dir_col = get_existing_col(df, [
        "Wave_direct", "Wave Direction", "Direction", "Dir"
    ])

    # ========================================================
    # 컬럼 확인 표시
    # ========================================================
    with st.expander("사용 컬럼 확인"):
        st.write("시간 컬럼:", time_col)
        st.write("온도 컬럼:", temp_col)
        st.write("파주기 컬럼:", period_col)
        st.write("유의파고 컬럼:", sig_wave_col)
        st.write("파고 컬럼:", wave_col)
        st.write("파향 컬럼:", dir_col)

    # ========================================================
    # 시간 처리
    # ========================================================
    if time_col is None:
        st.error("시간 컬럼을 찾지 못했습니다. (예: Time and Date)")
        st.stop()

    raw_time_sample = df[time_col].astype(str).head(10).tolist()

    df[time_col] = parse_datetime_series(df[time_col])
    invalid_time_count = df[time_col].isna().sum()

    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)

    with st.expander("시간 파싱 확인"):
        st.write("원본 시간 샘플:", raw_time_sample)
        st.write("변환 실패 개수:", int(invalid_time_count))
        st.write("최소 시간:", df[time_col].min())
        st.write("최대 시간:", df[time_col].max())

    # ========================================================
    # 숫자형 변환
    # ========================================================
    for col in [temp_col, period_col, sig_wave_col, wave_col, dir_col]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ========================================================
    # 필터 적용
    # ========================================================
    if wave_col is not None:
        df.loc[df[wave_col] > max_wave, wave_col] = np.nan

    if sig_wave_col is not None:
        df.loc[df[sig_wave_col] > max_sig_wave, sig_wave_col] = np.nan

    if period_col is not None:
        df.loc[df[period_col] > max_period, period_col] = np.nan

    # ========================================================
    # 데이터 정보
    # ========================================================
    st.subheader("📊 데이터 정보")

    avg_temp = df[temp_col].mean() if temp_col is not None else np.nan
    avg_period = df[period_col].mean() if period_col is not None else np.nan
    avg_sig_wave = df[sig_wave_col].mean() if sig_wave_col is not None else np.nan
    avg_wave = df[wave_col].mean() if wave_col is not None else np.nan
    avg_dir = circular_mean_deg(df[dir_col]) if dir_col is not None else np.nan

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:
        st.metric("데이터 개수", f"{len(df):,}")
    with c2:
        st.metric("평균 온도", "-" if pd.isna(avg_temp) else f"{avg_temp:.2f}")
    with c3:
        st.metric("평균 파주기", "-" if pd.isna(avg_period) else f"{avg_period:.2f}")

    with c4:
        st.metric("평균 파향", "-" if pd.isna(avg_dir) else f"{avg_dir:.2f}°")
    with c5:
        st.metric("평균 파고", "-" if pd.isna(avg_wave) else f"{avg_wave:.2f}")
    with c6:
        st.metric("평균 유의파고", "-" if pd.isna(avg_sig_wave) else f"{avg_sig_wave:.2f}")

    st.write(f"업로드 파일 수: **{len(uploaded_files)}개**")
    st.dataframe(
        pd.DataFrame({"파일명": [f.name for f in uploaded_files]}),
        use_container_width=True
    )

    # ========================================================
    # 10분 원형평균 파향 + 10분 평균 유의파고 (Wave Rose용)
    # ========================================================
    sea_rose = None

    if dir_col is not None and sig_wave_col is not None:
        df_rose = df[[time_col, dir_col, sig_wave_col]].copy()
        df_rose = df_rose.dropna(subset=[time_col])
        df_rose = df_rose.sort_values(time_col)
        df_rose = df_rose.set_index(time_col)

        dir_10 = df_rose[dir_col].resample("10min").apply(circular_mean_deg)
        hs_10 = df_rose[sig_wave_col].resample("10min").mean()

        sea_rose = pd.DataFrame({
            "MeanDir_deg": dir_10,
            "Hs_10": hs_10
        }).dropna()

    # ========================================================
    # 그래프
    # ========================================================
    st.subheader("📈 그래프")

    # 1) 온도
    if temp_col is not None:
        st.markdown("### 1) 온도 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[temp_col])
        plt.title("Temperature")
        plt.xlabel("Time")
        plt.ylabel(temp_col)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 2) 파주기
    if period_col is not None:
        st.markdown("### 2) 파주기 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[period_col])
        plt.title("Wave Period")
        plt.xlabel("Time")
        plt.ylabel(period_col)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 3) 유의파고
    if sig_wave_col is not None:
        st.markdown("### 3) 유의파고 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[sig_wave_col])
        plt.title("Significant Wave Height")
        plt.xlabel("Time")
        plt.ylabel(sig_wave_col)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 4) 파고
    if wave_col is not None:
        st.markdown("### 4) 파고 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[wave_col])
        plt.title("Wave Height")
        plt.xlabel("Time")
        plt.ylabel(wave_col)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 5) 파향
    if dir_col is not None:
        st.markdown("### 5) 파향 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[dir_col])
        plt.title("Wave Direction")
        plt.xlabel("Time")
        plt.ylabel(dir_col)
        plt.ylim(0, 360)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 6) Wave Rose
    if sea_rose is not None and len(sea_rose) > 0:
        st.markdown("### 6) Wave Rose (10분 원형평균 파향 기준)")

        try:
            from windrose import WindroseAxes

            fig_r = plt.figure(figsize=(8, 8))
            ax_r = WindroseAxes.from_ax(fig=fig_r)

            hmax = sea_rose["Hs_10"].max()

            if pd.isna(hmax) or hmax <= 0:
                st.warning("Wave Rose를 그릴 유의파고 데이터가 부족합니다.")
            else:
                bins = np.linspace(0, hmax, 6)

                ax_r.bar(
                    sea_rose["MeanDir_deg"],
                    sea_rose["Hs_10"],
                    normed=True,
                    opening=0.8,
                    edgecolor="white",
                    bins=bins
                )

                ax_r.set_title("Wave Rose (10-min Circular Mean Direction + Hs)")
                ax_r.set_legend(title="Hs")
                st.pyplot(fig_r)

        except Exception as e:
            st.warning(f"Wave Rose 생성 실패: {e}")

    else:
        st.info("Wave Rose를 그릴 수 있는 파향/유의파고 데이터가 부족합니다.")

    # ========================================================
    # 데이터 미리보기
    # ========================================================
    st.subheader("📋 데이터 미리보기")
    show_cols = [
        c for c in [time_col, temp_col, period_col, sig_wave_col, wave_col, dir_col, "source_file"]
        if c is not None
    ]
    st.dataframe(df[show_cols].head(50), use_container_width=True)

else:
    st.info("CSV 파일을 한 개 이상 업로드해 주세요.")
