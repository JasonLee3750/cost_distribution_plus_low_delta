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
# 1. 页面配置与 CSS 优化
# ===========================
st.set_page_config(
    page_title="全能交易控制台 (Streamlit版)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 让表格更好看
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
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

# 使用 st.cache_data 缓存数据，避免每次点击按钮都重新下载
@st.cache_data(ttl=3600)
def fetch_stock_history(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 布林带计算
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['SMA20'] + (2 * df['StdDev'])
    df['LowerBB'] = df['SMA20'] - (2 * df['StdDev'])
    
    return df

@st.cache_data(ttl=600) # 期权数据缓存时间短一点
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

        # 获取期权链日期
        exps = stock.options
        if not exps: return None, "无期权数据"
        
        target_date = datetime.now() + timedelta(days=days_expiry)
        closest_date = min(exps, key=lambda x: abs(datetime.strptime(x, "%Y-%m-%d") - target_date))
        real_expiry = datetime.strptime(closest_date, "%Y-%m-%d")
        real_days = (real_expiry - datetime.now()).days
        if real_days < 1: real_days = 1
        
        chain = stock.option_chain(closest_date).calls
        
        # ATM IV
        atm_contract = chain.iloc[(chain['strike'] - current_price).abs().argsort()[:1]]
        atm_iv = atm_contract['impliedVolatility'].values[0] * 100 if not atm_contract.empty else 0
        
        # 筛选：现价到现价*1.5
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
            
            res_item = {
                "行权价": k,
                "距现价(%)": (k - current_price) / current_price * 100,
                "IV(%)": iv * 100,
                "Mid价格": mid,
                "Delta": delta,
                "Gamma": gamma,
                "保留概率(%)": prob,
                "压力概率(%)": (1 - delta_stress) * 100,
                "加速风险(%)": gamma * (current_price * 0.01) * 100 * -1, # 存为正数方便显示，逻辑上还是绝对值
                "年化(%)": (mid / current_price) * (365 / real_days) * 100
            }
            results.append(res_item)
            
        info = {
            "S0": current_price, "ATM_IV": atm_iv, "HV5": hv5, "HV20": hv20,
            "expiry": closest_date, "days": real_days
        }
        return pd.DataFrame(results), info
    except Exception as e:
        return None, str(e)

# ===========================
# 3. Sidebar 全局设置
# ===========================
st.sidebar.title("⚙️ 设置面板")
ticker = st.sidebar.text_input("股票代码", value="TSLA").upper()

with st.sidebar.expander("📖 帮助文档"):
    st.markdown("""
    **1. 筹码分布 (Volume Profile):**
    右侧横向的彩色柱状图代表该价格区间的历史成交量。长条代表强支撑或强阻力。
    
    **2. 交互图表:**
    使用 Plotly 引擎，支持鼠标悬停查看价格、缩放和平移。
    
    **3. 卖方指标:**
    * **保留概率**: 到期不被行权的概率。
    * **加速风险**: 股价涨1%，胜率掉多少。越小越安全。
    * **IV vs HV**: 当 IV > HV 时，期权包含恐慌溢价，适合卖出。
    """)

# ===========================
# 4. 主界面 Logic
# ===========================
st.title(f"📈 全能交易控制台: {ticker}")

# 创建 Tabs
tab1, tab2 = st.tabs(["📊 筹码与趋势 (Charts)", "💰 期权卖方 (Option Seller)"])

# --------------------------
# Tab 1: 股价与筹码分布
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
            # --- 使用 Plotly 绘图 (完美交互) ---
            
            # 1. 计算筹码分布
            price_min, price_max = df['Close'].min(), df['Close'].max()
            bins = 80
            hist, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 2. 创建双轴图表
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 添加布林带 (填充)
            fig.add_trace(go.Scatter(x=df.index, y=df['UpperBB'], line=dict(color='rgba(0,128,0,0.3)', width=1), showlegend=False, hoverinfo='skip'), secondary_y=False)
            fig.add_trace(go.Scatter(x=df.index, y=df['LowerBB'], line=dict(color='rgba(0,128,0,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,128,0,0.05)', showlegend=False, hoverinfo='skip'), secondary_y=False)
            
            # 添加股价线
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='black', width=1.5)), secondary_y=False)
            
            # 添加筹码分布 (水平柱状图，挂在右轴或顶部轴，这里用简单的叠加模拟)
            # 为了不遮挡K线，我们将筹码分布画在左侧或作为背景，这里使用辅助X轴
            fig.add_trace(go.Bar(
                y=bin_centers, 
                x=hist, 
                orientation='h', 
                name='Volume Profile',
                marker=dict(color='orange', opacity=0.3),
                xaxis='x2', # 使用第二个X轴
                hoverinfo='none'
            ))

            # 解析目标价
            targets = []
            if target_str:
                try:
                    targets = [float(x) for x in target_str.replace('，', ',').split(',') if x.strip()][:3]
                except: pass

            # 添加横线 (现价 & 目标价)
            current_price = df['Close'].iloc[-1]
            fig.add_hline(y=current_price, line_dash="dash", line_color="red", annotation_text=f"Current: {current_price:.2f}")
            
            line_styles = ["dashdot", "dot", "dash"]
            for i, t_price in enumerate(targets):
                fig.add_hline(y=t_price, line_dash=line_styles[i%3], line_color="blue", annotation_text=f"Target: {t_price:.2f}")

            # 布局设置
            fig.update_layout(
                title=f"{ticker} Price & Volume Profile",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Price"),
                xaxis2=dict(
                    title="Volume Profile", 
                    overlaying="x", 
                    side="top", 
                    showgrid=False, 
                    showticklabels=False,
                    range=[0, max(hist)*3] # 让柱子只占 1/3 宽度，不遮挡股价
                ),
                height=600,
                hovermode="x unified", # 完美的交互光标
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("无法获取股票数据，请检查代码。")

# --------------------------
# Tab 2: 期权卖方扫描器
# --------------------------
with tab2:
    # 第一行：输入参数
    c1, c2 = st.columns([1, 4])
    with c1:
        days = st.number_input("到期天数", min_value=1, value=7, step=1)
    
    # 第二行：阈值筛选 (放在 Expander 里保持整洁)
    with st.expander("🚦 筛选阈值设置 (绿灯条件)", expanded=True):
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        th_prob = col_t1.number_input("保留概率 >", value=90.0, step=1.0)
        th_stress = col_t2.number_input("IV+10%概率 >", value=80.0, step=1.0)
        th_gamma = col_t3.number_input("加速风险 > (绝对值)", value=-1.5, step=0.1) # 输入负数比较麻烦，逻辑上这里输入界限
        th_apr = col_t4.number_input("年化收益 >", value=10.0, step=1.0)

    if st.button("🔍 扫描期权链", type="primary"):
        with st.spinner("正在计算 Greeks 和 BSM 模型..."):
            df_opt, info = fetch_option_chain(ticker, days)
        
        if isinstance(info, str):
            st.error(info)
        else:
            # 1. 显示市场概况 Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("当前股价", f"${info['S0']:.2f}")
            m2.metric("到期日", f"{info['expiry']}", f"剩余 {info['days']} 天")
            
            # IV 状态判断
            iv_delta = info['ATM_IV'] - info['HV5']
            iv_color = "normal"
            if iv_delta > 5: iv_color = "inverse" # 绿色
            elif iv_delta < -5: iv_color = "off"
            
            m3.metric("ATM IV", f"{info['ATM_IV']:.1f}%", delta=f"{iv_delta:.1f}% vs HV5")
            m4.metric("5日历史波动 (HV5)", f"{info['HV5']:.1f}%")

            if iv_delta > 5:
                st.success("🔥 IV 显著高于 HV，恐慌溢价较高，适合卖出！")
            elif iv_delta < -5:
                st.warning("🧊 IV 低于 HV，期权便宜，卖出肉少。")

            # 2. 数据处理与高亮
            # 逻辑修正：加速风险在 DataFrame 里是 gamma * price * 0.01 * 100 * -1 (存为了正值方便看?) 
            # 让我们看上面的 fetch_option_chain 实现:
            # "加速风险(%)": gamma * ... * -1. 
            # 所以如果 th_gamma 输入 -1.5，我们希望风险显示值 > -1.5 (比如 -0.5)。
            # 为了方便用户，显示的时候通常显示负数。
            
            # 重新调整 DataFrame 用于显示的 Gamma 风险为负数 (因为上面 * -1 了)
            df_opt["加速风险(%)"] = df_opt["加速风险(%)"] * -1

            def highlight_perfect(row):
                # 筛选逻辑
                is_perfect = (
                    row['保留概率(%)'] >= th_prob and
                    row['压力概率(%)'] >= th_stress and
                    row['加速风险(%)'] >= th_gamma and # 比如 -0.5 >= -1.5 (True)
                    row['年化(%)'] >= th_apr
                )
                if is_perfect:
                    return ['background-color: #d4edda; color: green'] * len(row)
                return [''] * len(row)

            # 3. 展示表格
            st.dataframe(
                df_opt.style.apply(highlight_perfect, axis=1).format({
                    "距现价(%)": "{:.1f}%",
                    "IV(%)": "{:.1f}%",
                    "Mid价格": "${:.2f}",
                    "Delta": "{:.3f}",
                    "Gamma": "{:.4f}",
                    "保留概率(%)": "{:.1f}%",
                    "压力概率(%)": "{:.1f}%",
                    "加速风险(%)": "{:.2f}%",
                    "年化(%)": "{:.1f}%"
                }),
                use_container_width=True,
                height=600
            )