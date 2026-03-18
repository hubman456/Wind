import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 기본 설정
# ============================================================
st.set_page_config(page_title="Wind Monitoring Dashboard", layout="wide")
st.title("🌬 Wind Monitoring Dashboard")

# ============================================================
# 사이드바 필터
# ============================================================
with st.sidebar:
    st.header("⚙ 필터 설정")

    max_wind_speed = st.number_input("Wind Speed 최대값", value=40.0)
    min_wind_speed = st.number_input("Wind Speed 최소값", value=0.0)

    use_temp_filter = st.checkbox("온도 필터 사용", value=False)
    min_temp = st.number_input("최소 온도", value=-20.0)
    max_temp = st.number_input("최대 온도", value=50.0)

    use_ti_filter = st.checkbox("난류강도 필터 사용", value=False)
    max_ti = st.number_input("난류강도 최대값", value=1.0)

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
    """풍향 평균용 원형평균"""
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
    날짜 파싱을 비교적 안전하게 수행
    """
    s = series.astype(str).str.strip()

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

    best_dt = None
    best_valid_count = -1

    for fmt in known_formats:
        try:
            dt_try = pd.to_datetime(s, format=fmt, errors="coerce")
            valid_count = dt_try.notna().sum()

            if valid_count > best_valid_count:
                best_valid_count = valid_count
                best_dt = dt_try
        except Exception:
            pass

    if best_dt is None or best_valid_count == 0:
        best_dt = pd.to_datetime(s, errors="coerce")

    return best_dt

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

    speed_col = get_existing_col(df, [
        "Wind Speed", "WS", "WindSpeed", "Mean_WS", "WS_mean",
        "Horizontal Wind Speed", "Avg Wind Speed"
    ])

    dir_col = get_existing_col(df, [
        "Wind Direction", "WD", "WindDir", "Mean_WD", "WD_mean",
        "Avg Wind Direction"
    ])

    temp_col = get_existing_col(df, [
        "Temperature", "Temp", "temperature", "Air Temperature"
    ])

    ti_col = get_existing_col(df, [
        "TI", "Turbulence Intensity", "TI_mean", "Mean_TI"
    ])

    # ========================================================
    # 컬럼 확인
    # ========================================================
    with st.expander("사용 컬럼 확인"):
        st.write("시간 컬럼:", time_col)
        st.write("풍속 컬럼:", speed_col)
        st.write("풍향 컬럼:", dir_col)
        st.write("온도 컬럼:", temp_col)
        st.write("난류강도 컬럼:", ti_col)

    if time_col is None:
        st.error("시간 컬럼을 찾지 못했습니다. (예: Time and Date)")
        st.stop()

    # ========================================================
    # 시간 처리
    # ========================================================
    raw_time_sample = df[time_col].astype(str).head(10).tolist()

    df[time_col] = parse_datetime_series(df[time_col])
    invalid_time_count = df[time_col].isna().sum()

    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)

    with st.expander("시간 파싱 확인", expanded=False):
        st.write("원본 시간 샘플:", raw_time_sample)
        st.write("변환 실패 개수:", int(invalid_time_count))
        st.write("최소 시간:", df[time_col].min())
        st.write("최대 시간:", df[time_col].max())

    # ========================================================
    # 숫자형 변환
    # ========================================================
    for col in [speed_col, dir_col, temp_col, ti_col]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ========================================================
    # 필터 적용
    # ========================================================
    if speed_col is not None:
        df.loc[df[speed_col] > max_wind_speed, speed_col] = np.nan
        df.loc[df[speed_col] < min_wind_speed, speed_col] = np.nan

    if use_temp_filter and temp_col is not None:
        df.loc[(df[temp_col] < min_temp) | (df[temp_col] > max_temp), temp_col] = np.nan

    if use_ti_filter and ti_col is not None:
        df.loc[df[ti_col] > max_ti, ti_col] = np.nan

    # ========================================================
    # 데이터 정보
    # ========================================================
    st.subheader("📊 데이터 정보")

    avg_speed = df[speed_col].mean() if speed_col is not None else np.nan
    max_speed = df[speed_col].max() if speed_col is not None else np.nan
    avg_dir = circular_mean_deg(df[dir_col]) if dir_col is not None else np.nan
    avg_temp = df[temp_col].mean() if temp_col is not None else np.nan
    avg_ti = df[ti_col].mean() if ti_col is not None else np.nan

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:
        st.metric("데이터 개수", f"{len(df):,}")
    with c2:
        st.metric("평균 풍속", "-" if pd.isna(avg_speed) else f"{avg_speed:.2f}")
    with c3:
        st.metric("최대 풍속", "-" if pd.isna(max_speed) else f"{max_speed:.2f}")

    with c4:
        st.metric("평균 풍향", "-" if pd.isna(avg_dir) else f"{avg_dir:.2f}°")
    with c5:
        st.metric("평균 온도", "-" if pd.isna(avg_temp) else f"{avg_temp:.2f}")
    with c6:
        st.metric("평균 난류강도", "-" if pd.isna(avg_ti) else f"{avg_ti:.3f}")

    st.write(f"업로드 파일 수: **{len(uploaded_files)}개**")
    st.dataframe(
        pd.DataFrame({"파일명": [f.name for f in uploaded_files]}),
        use_container_width=True
    )

    # ========================================================
    # Wind Rose용 10분 평균 데이터
    # - 방향: 10분 원형평균
    # - 풍속: 10분 평균
    # ========================================================
    wind_rose = None

    if dir_col is not None and speed_col is not None:
        df_rose = df[[time_col, dir_col, speed_col]].copy()
        df_rose = df_rose.dropna(subset=[time_col])
        df_rose = df_rose.sort_values(time_col)
        df_rose = df_rose.set_index(time_col)

        dir_10 = df_rose[dir_col].resample("10min").apply(circular_mean_deg)
        speed_10 = df_rose[speed_col].resample("10min").mean()

        wind_rose = pd.DataFrame({
            "MeanDir_deg": dir_10,
            "WS_10": speed_10
        }).dropna()

    # ========================================================
    # 그래프
    # ========================================================
    st.subheader("📈 그래프")

    # 1) 풍속
    if speed_col is not None:
        st.markdown("### 1) 풍속 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[speed_col])
        plt.title("Wind Speed")
        plt.xlabel("Time")
        plt.ylabel(speed_col)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 2) 풍향
    if dir_col is not None:
        st.markdown("### 2) 풍향 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[dir_col])
        plt.title("Wind Direction")
        plt.xlabel("Time")
        plt.ylabel(dir_col)
        plt.ylim(0, 360)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 3) 온도
    if temp_col is not None:
        st.markdown("### 3) 온도 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[temp_col])
        plt.title("Temperature")
        plt.xlabel("Time")
        plt.ylabel(temp_col)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 4) 난류강도
    if ti_col is not None:
        st.markdown("### 4) 난류강도 그래프")
        fig = plt.figure(figsize=(12, 4))
        plt.plot(df[time_col], df[ti_col])
        plt.title("Turbulence Intensity")
        plt.xlabel("Time")
        plt.ylabel(ti_col)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # 5) Wind Rose
    if wind_rose is not None and len(wind_rose) > 0:
        st.markdown("### 5) Wind Rose (10분 원형평균 풍향 기준)")

        try:
            from windrose import WindroseAxes

            fig_r = plt.figure(figsize=(8, 8))
            ax_r = WindroseAxes.from_ax(fig=fig_r)

            ws_max = wind_rose["WS_10"].max()

            if pd.isna(ws_max) or ws_max <= 0:
                st.warning("Wind Rose를 그릴 풍속 데이터가 부족합니다.")
            else:
                bins = np.linspace(0, ws_max, 6)

                ax_r.bar(
                    wind_rose["MeanDir_deg"],
                    wind_rose["WS_10"],
                    normed=True,
                    opening=0.8,
                    edgecolor="white",
                    bins=bins
                )

                ax_r.set_title("Wind Rose (10-min Circular Mean Direction + Wind Speed)")
                ax_r.set_legend(title="Wind Speed")
                st.pyplot(fig_r)

        except Exception as e:
            st.warning(f"Wind Rose 생성 실패: {e}")
    else:
        st.info("Wind Rose를 그릴 수 있는 풍향/풍속 데이터가 부족합니다.")

    # ========================================================
    # 데이터 미리보기
    # ========================================================
    st.subheader("📋 데이터 미리보기")
    show_cols = [
        c for c in [time_col, speed_col, dir_col, temp_col, ti_col, "source_file"]
        if c is not None
    ]
    st.dataframe(df[show_cols].head(50), use_container_width=True)

else:
    st.info("CSV 파일을 한 개 이상 업로드해 주세요.")
