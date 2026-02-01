import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ===========================
# 1. 页面配置与 CSS
# ===========================
st.set_page_config(
    page_title="全能交易控制台 (Streamlit版)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    /* 调整复选框的样式，让它更醒目 */
    div[data-testid="stCheckbox"] label {
        font-weight: bold;
        color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# 2. 数学核心 (BSM & Greeks)
# ===========================
def bsm_calc(S, K, T, r, sigma):
    if T <= 0: return 0, 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return price, delta, gamma

@st.cache_data(ttl=3600)
def fetch_stock_history(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 布林带
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['SMA20'] + (2 * df['StdDev'])
    df['LowerBB'] = df['SMA20'] - (2 * df['StdDev'])
    
    return df

@st.cache_data(ttl=600)
def fetch_option_chain(ticker, days_expiry):
    try:
        stock = yf.Ticker(ticker)
        # 获取股价与波动率
        hist = stock.history(period="1mo")
        if hist.empty: return None, "无法获取股价"
        current_price = hist['Close'].iloc[-1]
        
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
        hv5 = log_ret.tail(5).std() * np.sqrt(252) * 100
        hv20 = log_ret.tail(20).std() * np.sqrt(252) * 100

        # 获取期权链
        exps = stock.options
        if not exps: return None, "无期权数据"
        
        target_date = datetime.now() + timedelta(days=days_expiry)
        closest_date = min(exps, key=lambda x: abs(datetime.strptime(x, "%Y-%m-%d") - target_date))
        real_expiry = datetime.strptime(closest_date, "%Y-%m-%d")
        real_days = (real_expiry - datetime.now()).days
        if real_days < 1: real_days = 1
        
        # --- 新增：计算星期几 ---
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday_str = weekdays[real_expiry.weekday()]
        expiry_display = f"{closest_date} ({weekday_str})"
        # ---------------------
        
        chain = stock.option_chain(closest_date).calls
        
        # ATM IV
        atm_contract = chain.iloc[(chain['strike'] - current_price).abs().argsort()[:1]]
        atm_iv = atm_contract['impliedVolatility'].values[0] * 100 if not atm_contract.empty else 0
        
        # 筛选
        max_strike = current_price * 1.5
        chain = chain[(chain['strike'] > current_price) & (chain['strike'] <= max_strike) & (chain['impliedVolatility'] > 0.001)].copy()
        
        results = []
        r = 0.045
        T_yrs = real_days / 365.0
        
        for _, row in chain.iterrows():
            k = row['strike']
            mid = (row['bid'] + row['ask']) / 2 if row['ask'] > 0 else row['lastPrice']
            iv = row['impliedVolatility']
            
            _, delta, gamma = bsm_calc(current_price, k, T_yrs, r, iv)
            prob = (1 - delta) * 100
            
            if prob < 30: continue
            
            _, delta_stress, _ = bsm_calc(current_price, k, T_yrs, r, iv + 0.10)
            
            gamma_risk = gamma * (current_price * 0.01) * 100 * -1
            
            res_item = {
                "行权价": k,
                "距现价(%)": (k - current_price) / current_price * 100,
                "IV(%)": iv * 100,
                "Mid价格": mid,
                "Delta": delta,
                "Gamma": gamma,
                "保留概率(%)": prob,
                "压力概率(%)": (1 - delta_stress) * 100,
                "加速风险(%)": gamma_risk, 
                "年化(%)": (mid / current_price) * (365 / real_days) * 100
            }
            results.append(res_item)
            
        info = {
            "S0": current_price, "ATM_IV": atm_iv, "HV5": hv5, "HV20": hv20,
            "expiry": expiry_display, # 这里现在是带星期的字符串了
            "days": real_days
        }
        return pd.DataFrame(results), info
    except Exception as e:
        return None, str(e)

# ===========================
# 3. Sidebar 设置
# ===========================
st.sidebar.title("⚙️ 设置面板")
ticker = st.sidebar.text_input("股票代码", value="TSLA").upper()

with st.sidebar.expander("📖 帮助文档"):
    st.markdown("""
    **卖方指标说明:**
    * **保留概率**: 建议 > 90%。
    * **加速风险**: 股价涨1%时胜率下降的幅度。这是负数，例如 -0.5% 比 -2.0% 更安全。设置阈值时输入如 -1.5，代表只接受 > -1.5 (即衰减更小) 的合约。
    * **IV vs HV**: 当 IV > HV 时适合卖出。
    """)

# ===========================
# 4. 主界面 Logic
# ===========================
st.title(f"📈 全能交易控制台: {ticker}")

tab1, tab2 = st.tabs(["📊 筹码与趋势 (Charts)", "💰 期权卖方 (Option Seller)"])

# --------------------------
# Tab 1: 股价与筹码
# --------------------------
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        period = st.selectbox("时间范围", ["3mo", "6mo", "1y", "2y"], index=1)
    with col2:
        target_str = st.text_input("目标价 (蓝线，逗号隔开)", value="240, 250")
    
    if ticker:
        df = fetch_stock_history(ticker, period)
        if df is not None:
            price_min, price_max = df['Close'].min(), df['Close'].max()
            bins = 80
            hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 布林带
            fig.add_trace(go.Scatter(x=df.index, y=df['UpperBB'], line=dict(color='rgba(0,128,0,0.3)', width=1), showlegend=False, hoverinfo='skip'), secondary_y=False)
            fig.add_trace(go.Scatter(x=df.index, y=df['LowerBB'], line=dict(color='rgba(0,128,0,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,128,0,0.05)', showlegend=False, hoverinfo='skip'), secondary_y=False)
            
            # 股价
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='black', width=1.5)), secondary_y=False)
            
            # 筹码
            fig.add_trace(go.Bar(y=bin_centers, x=hist, orientation='h', name='Volume Profile', marker=dict(color='orange', opacity=0.3), xaxis='x2', hoverinfo='none'))

            # 辅助线
            targets = []
            if target_str:
                try: targets = [float(x) for x in target_str.replace('，', ',').split(',') if x.strip()][:3]
                except: pass

            current_price = df['Close'].iloc[-1]
            fig.add_hline(y=current_price, line_dash="dash", line_color="red", annotation_text=f"Current: {current_price:.2f}")
            
            styles = ["dashdot", "dot", "dash"]
            for i, t in enumerate(targets):
                fig.add_hline(y=t, line_dash=styles[i%3], line_color="blue", annotation_text=f"Target: {t:.2f}")

            fig.update_layout(
                title=f"{ticker} Price & Volume Profile",
                xaxis=dict(title="Date"), yaxis=dict(title="Price"),
                xaxis2=dict(title="Volume", overlaying="x", side="top", showgrid=False, showticklabels=False, range=[0, max(hist)*3]),
                height=600, hovermode="x unified", template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("数据获取失败")

# --------------------------
# Tab 2: 期权卖方 (修复版)
# --------------------------
with tab2:
    # 1. 参数输入
    c1, c2 = st.columns([1, 4])
    with c1:
        days = st.number_input("到期天数", min_value=1, value=7, step=1)
    
    # 2. 筛选器
    with st.expander("🚦 筛选阈值设置 (绿灯条件)", expanded=True):
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        th_prob = col_t1.number_input("保留概率 >", value=90.0, step=1.0)
        th_stress = col_t2.number_input("IV+10%概率 >", value=80.0, step=1.0)
        th_gamma = col_t3.number_input("加速风险 > (通常为负数)", value=-1.5, step=0.1, help="例如: -0.5 > -1.5，表示风险更小")
        th_apr = col_t4.number_input("年化收益 >", value=10.0, step=1.0)

    # 3. 过滤开关
    show_only_perfect = st.checkbox("☑️ 只显示符合条件的完美合约 (Only Show Perfect Matches)", value=False)

    if st.button("🔍 扫描期权链", type="primary"):
        with st.spinner("计算中..."):
            df_opt, info = fetch_option_chain(ticker, days)
        
        if isinstance(info, str):
            st.error(info)
        else:
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("当前股价", f"${info['S0']:.2f}")
            m2.metric("到期", f"{info['expiry']}", f"剩余 {info['days']} 天")
            iv_diff = info['ATM_IV'] - info['HV5']
            m3.metric("ATM IV", f"{info['ATM_IV']:.1f}%", f"{iv_diff:.1f}% vs HV5")
            m4.metric("5日HV", f"{info['HV5']:.1f}%")

            if iv_diff > 5: st.success("🔥 IV 较高，适合卖出！")
            elif iv_diff < -5: st.warning("🧊 IV 较低，肉少风险大。")

            # --- 核心逻辑: 标记与筛选 ---
            def check_perfect(row):
                return (
                    row['保留概率(%)'] >= th_prob and
                    row['压力概率(%)'] >= th_stress and
                    row['加速风险(%)'] >= th_gamma and 
                    row['年化(%)'] >= th_apr
                )

            # 生成布尔列
            df_opt['Is_Perfect'] = df_opt.apply(check_perfect, axis=1)

            # 根据开关决定显示哪些数据
            if show_only_perfect:
                df_display = df_opt[df_opt['Is_Perfect']].copy()
                if df_display.empty:
                    st.warning("⚠️ 没有找到符合条件的完美合约。")
            else:
                df_display = df_opt.copy()

            # --- 修复点：定义样式函数 ---
            def highlight_rows(row):
                # 这里 row 包含了 Is_Perfect，所以不会报错了
                if row['Is_Perfect']:
                    return ['background-color: #d4edda; color: green'] * len(row)
                return [''] * len(row)

            # --- 修复点：先创建 Styler，最后隐藏列 ---
            # 1. 应用样式
            styler = df_display.style.apply(highlight_rows, axis=1)
            
            # 2. 格式化数字
            styler = styler.format({
                "距现价(%)": "{:.1f}%",
                "IV(%)": "{:.1f}%",
                "Mid价格": "${:.2f}",
                "Delta": "{:.3f}",
                "Gamma": "{:.4f}",
                "保留概率(%)": "{:.1f}%",
                "压力概率(%)": "{:.1f}%",
                "加速风险(%)": "{:.2f}%",
                "年化(%)": "{:.1f}%"
            })
            
            # 3. 隐藏 'Is_Perfect' 辅助列 (Streamlit 会识别这个设置)
            # 兼容不同版本的 Pandas
            try:
                styler = styler.hide(axis="columns", subset=["Is_Perfect"])
            except AttributeError:
                # 老版本 Pandas 写法
                styler = styler.hide_columns(["Is_Perfect"])

            # 4. 显示最终表格
            st.dataframe(
                styler,
                use_container_width=True,
                height=600
            )
            
            # 底部计数
            count = df_opt['Is_Perfect'].sum()
            st.caption(f"共扫描 {len(df_opt)} 个合约，其中 {count} 个完美符合条件。")
