import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Wind Monitoring Dashboard", layout="wide")
st.title("🌬 Wind Monitoring Dashboard")

# ============================================================
# 유틸 함수
# ============================================================
def circular_mean_deg(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan

    rad = np.deg2rad(s)
    mean_sin = np.sin(rad).mean()
    mean_cos = np.cos(rad).mean()
    angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    return (angle + 360) % 360


def parse_time_series(series):
    s = series.astype(str).str.strip()

    # 1차: 일반 파싱
    dt = pd.to_datetime(s, errors="coerce")

    # 2차: dayfirst 재시도
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(s[mask], errors="coerce", dayfirst=True)
        dt.loc[mask] = dt2

    return dt


def extract_available_heights(columns):
    heights = set()

    patterns = [
        r"Wind Direction \(deg\) at (\d+)m \(corrected\)",
        r"Horizontal Wind Speed \(m/s\) at (\d+)m",
        r"TI at (\d+)m",
    ]

    for col in columns:
        for p in patterns:
            m = re.search(p, col)
            if m:
                heights.add(int(m.group(1)))

    return sorted(heights, reverse=True)


def get_col_for_height(height):
    wd_col = f"Wind Direction (deg) at {height}m (corrected)"
    ws_col = f"Horizontal Wind Speed (m/s) at {height}m"
    ti_col = f"TI at {height}m"
    return wd_col, ws_col, ti_col


# ============================================================
# 파일 업로드
# ============================================================
uploaded_files = st.file_uploader(
    "CSV 파일 업로드 (여러 개 가능)",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("CSV 파일을 한 개 이상 업로드해 주세요.")
    st.stop()

# ============================================================
# CSV 읽기
# ============================================================
dfs = []
for f in uploaded_files:
    try:
        df_temp = pd.read_csv(f)
        df_temp.columns = df_temp.columns.str.strip()
        df_temp["source_file"] = f.name
        dfs.append(df_temp)
    except Exception as e:
        st.warning(f"{f.name} 읽기 실패: {e}")

if len(dfs) == 0:
    st.error("읽을 수 있는 CSV 파일이 없습니다.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# ============================================================
# 기본 컬럼 지정
# ============================================================
time_col = "Time and Date"
temp_col = "Met Air Temp. (C)"

if time_col not in df.columns:
    st.error(f"시간 컬럼을 찾지 못했습니다: {time_col}")
    st.write("현재 컬럼명:", list(df.columns))
    st.stop()

if temp_col not in df.columns:
    temp_col = None

# ============================================================
# 높이 추출
# ============================================================
available_heights = extract_available_heights(df.columns)

if len(available_heights) == 0:
    st.error("풍속/풍향/TI 높이 컬럼을 찾지 못했습니다.")
    st.write("현재 컬럼명:", list(df.columns))
    st.stop()

# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.header("⚙ 필터 설정")

    selected_height = st.selectbox(
        "분석 높이 선택",
        available_heights,
        index=0
    )

    max_ws = st.number_input("Wind Speed 최대값", value=40.0)
    min_ws = st.number_input("Wind Speed 최소값", value=0.0)

    use_temp_filter = st.checkbox("온도 필터 사용", value=False)
    min_temp = st.number_input("최소 온도", value=-30.0)
    max_temp = st.number_input("최대 온도", value=50.0)

    use_ti_filter = st.checkbox("난류강도 필터 사용", value=False)
    max_ti = st.number_input("난류강도 최대값", value=1.0)

# 선택 높이 컬럼명
wd_col, ws_col, ti_col = get_col_for_height(selected_height)

missing_cols = [c for c in [wd_col, ws_col, ti_col] if c not in df.columns]
if missing_cols:
    st.error(f"선택 높이({selected_height}m)의 컬럼이 없습니다: {missing_cols}")
    st.stop()

# ============================================================
# 시간 처리
# ============================================================
raw_time_sample = df[time_col].astype(str).head(10).tolist()

df[time_col] = parse_time_series(df[time_col])
invalid_time_count = df[time_col].isna().sum()

df = df.dropna(subset=[time_col]).sort_values(time_col)

# ============================================================
# 숫자형 변환
# ============================================================
numeric_cols = [ws_col, wd_col, ti_col]
if temp_col is not None:
    numeric_cols.append(temp_col)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# 필터 적용
# ============================================================
df.loc[df[ws_col] > max_ws, ws_col] = np.nan
df.loc[df[ws_col] < min_ws, ws_col] = np.nan

if use_temp_filter and temp_col is not None:
    df.loc[(df[temp_col] < min_temp) | (df[temp_col] > max_temp), temp_col] = np.nan

if use_ti_filter:
    df.loc[df[ti_col] > max_ti, ti_col] = np.nan

# ============================================================
# 컬럼/시간 확인
# ============================================================
with st.expander("사용 컬럼 확인"):
    st.write("시간 컬럼:", time_col)
    st.write("온도 컬럼:", temp_col)
    st.write("풍속 컬럼:", ws_col)
    st.write("풍향 컬럼:", wd_col)
    st.write("난류강도 컬럼:", ti_col)

with st.expander("시간 파싱 확인"):
    st.write("원본 시간 샘플:", raw_time_sample)
    st.write("변환 실패 개수:", int(invalid_time_count))
    st.write("최소 시간:", df[time_col].min())
    st.write("최대 시간:", df[time_col].max())

# ============================================================
# 데이터 정보
# ============================================================
st.subheader(f"📊 데이터 정보 ({selected_height}m 기준)")

avg_ws = df[ws_col].mean()
max_ws_val = df[ws_col].max()
avg_wd = circular_mean_deg(df[wd_col])
avg_temp = df[temp_col].mean() if temp_col is not None else np.nan
avg_ti = df[ti_col].mean()

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

with c1:
    st.metric("데이터 개수", f"{len(df):,}")
with c2:
    st.metric("평균 풍속", "-" if pd.isna(avg_ws) else f"{avg_ws:.2f}")
with c3:
    st.metric("최대 풍속", "-" if pd.isna(max_ws_val) else f"{max_ws_val:.2f}")

with c4:
    st.metric("평균 풍향", "-" if pd.isna(avg_wd) else f"{avg_wd:.2f}°")
with c5:
    st.metric("평균 온도", "-" if pd.isna(avg_temp) else f"{avg_temp:.2f}")
with c6:
    st.metric("평균 난류강도", "-" if pd.isna(avg_ti) else f"{avg_ti:.3f}")

st.write(f"업로드 파일 수: **{len(uploaded_files)}개**")
st.dataframe(
    pd.DataFrame({"파일명": [f.name for f in uploaded_files]}),
    use_container_width=True
)

# ============================================================
# 그래프
# ============================================================
st.subheader(f"📈 그래프 ({selected_height}m 기준)")

# 1) 풍속
st.markdown("### 1) 풍속 그래프")
fig1 = plt.figure(figsize=(12, 4))
plt.plot(df[time_col], df[ws_col])
plt.title(f"Horizontal Wind Speed at {selected_height}m")
plt.xlabel("Time")
plt.ylabel("Wind Speed (m/s)")
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig1)

# 2) 풍향
st.markdown("### 2) 풍향 그래프")
fig2 = plt.figure(figsize=(12, 4))
plt.plot(df[time_col], df[wd_col])
plt.title(f"Wind Direction at {selected_height}m")
plt.xlabel("Time")
plt.ylabel("Wind Direction (deg)")
plt.ylim(0, 360)
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig2)

# 3) 온도
if temp_col is not None:
    st.markdown("### 3) 온도 그래프")
    fig3 = plt.figure(figsize=(12, 4))
    plt.plot(df[time_col], df[temp_col])
    plt.title("Met Air Temperature")
    plt.xlabel("Time")
    plt.ylabel("Temperature (C)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig3)

# 4) 난류강도
st.markdown("### 4) 난류강도 그래프")
fig4 = plt.figure(figsize=(12, 4))
plt.plot(df[time_col], df[ti_col])
plt.title(f"Turbulence Intensity at {selected_height}m")
plt.xlabel("Time")
plt.ylabel("TI")
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig4)

# ============================================================
# Wind Rose
# - 방향: 10분 원형평균
# - 풍속: 10분 평균
# ============================================================
st.markdown("### 5) Wind Rose (10분 원형평균 풍향 기준)")

try:
    from windrose import WindroseAxes

    df_r = df[[time_col, wd_col, ws_col]].dropna().copy()
    df_r = df_r.set_index(time_col)

    dir_10 = df_r[wd_col].resample("10min").apply(circular_mean_deg)
    ws_10 = df_r[ws_col].resample("10min").mean()

    rose = pd.DataFrame({
        "dir": dir_10,
        "ws": ws_10
    }).dropna()

    if len(rose) == 0:
        st.warning("Wind Rose를 그릴 데이터가 부족합니다.")
    else:
        fig_r = plt.figure(figsize=(8, 8))
        ax = WindroseAxes.from_ax(fig=fig_r)

        ws_max = rose["ws"].max()
        bins = np.linspace(0, ws_max, 6) if ws_max > 0 else [0, 1]

        ax.bar(
            rose["dir"],
            rose["ws"],
            normed=True,
            opening=0.8,
            edgecolor="white",
            bins=bins
        )

        ax.set_title(f"Wind Rose at {selected_height}m")
        ax.set_legend(title="Wind Speed")
        st.pyplot(fig_r)

except Exception as e:
    st.warning(f"Wind Rose 생성 실패: {e}")

# ============================================================
# 데이터 미리보기
# ============================================================
st.subheader("📋 데이터 미리보기")
preview_cols = [time_col, ws_col, wd_col, ti_col, "source_file"]
if temp_col is not None:
    preview_cols.insert(3, temp_col)

st.dataframe(df[preview_cols].head(50), use_container_width=True)
