import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="ETF均线回撤 + BS指标", layout="wide")
st.title("ETF 均线回撤策略 + BS 指标助手")

# ------------------- 输入参数 -------------------
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    code = st.text_input("输入ETF代码（如 510300、159915）", "510300")
    start = st.date_input("开始日期", pd.to_datetime("2023-01-01"))
    end = st.date_input("结束日期", pd.to_datetime(datetime.today()))

with col_input2:
    ma_short = st.number_input("短期均线天数", min_value=2, max_value=120, value=20)
    ma_long = st.number_input("长期均线天数", min_value=2, max_value=250, value=60)
    dd_pct = st.number_input("回撤阈值（%）", min_value=0.0, max_value=50.0, value=8.0)
    # BS指标参数
    rsi_period = st.number_input("RSI周期", min_value=5, max_value=30, value=14)
    vol_ma_period = st.number_input("成交量均线周期", min_value=3, max_value=30, value=5)

# ------------------- 数据获取 -------------------
@st.cache_data(ttl=600)
def load_etf_data(symbol, start_date, end_date):
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ------------------- 技术指标计算 -------------------
def add_technical_indicators(df, ma_short, ma_long, rsi_period, vol_ma_period):
    # 均线
    df[f"MA{ma_short}"] = df["close"].rolling(ma_short).mean()
    df[f"MA{ma_long}"] = df["close"].rolling(ma_long).mean()

    # 回撤
    df["rolling_high"] = df["close"].cummax()
    df["drawdown"] = (df["close"] - df["rolling_high"]) / df["rolling_high"] * 100

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = (df["DIF"] - df["DEA"]) * 2

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 成交量均线
    df[f"VOL_MA{vol_ma_period}"] = df["volume"].rolling(vol_ma_period).mean()

    return df

# ------------------- BS信号生成 -------------------
def generate_bs_signals(df, ma_short, dd_pct, vol_ma_period):
    """
    B点：收盘价 > MA_short，MACD金叉，RSI > 50，成交量放大
    S点：收盘价 < MA_short 或 MACD死叉 或 回撤超过阈值
    """
    # 初始化信号列
    df["bs_signal"] = ""  # 空字符串表示无信号
    df["bs_type"] = ""    # B 或 S

    # 条件
    price_above_ma = df["close"] > df[f"MA{ma_short}"]
    macd_golden_cross = (df["DIF"] > df["DEA"]) & (df["DIF"].shift(1) <= df["DEA"].shift(1))
    macd_dead_cross = (df["DIF"] < df["DEA"]) & (df["DIF"].shift(1) >= df["DEA"].shift(1))
    rsi_bullish = df["RSI"] > 50
    volume_expand = df["volume"] > df[f"VOL_MA{vol_ma_period}"]
    drawdown_exceed = df["drawdown"] <= -dd_pct

    # B点条件
    buy_cond = price_above_ma & macd_golden_cross & rsi_bullish & volume_expand

    # S点条件：死叉，或跌破均线且回撤超阈值，或回撤超阈值
    sell_cond = macd_dead_cross | (drawdown_exceed & ~price_above_ma) | drawdown_exceed

    df.loc[buy_cond, "bs_signal"] = "B"
    df.loc[buy_cond, "bs_type"] = "B"
    df.loc[sell_cond & (df["bs_signal"] == ""), "bs_signal"] = "S"
    df.loc[sell_cond & (df["bs_type"] == ""), "bs_type"] = "S"

    # 避免同一日同时出现 B 和 S，后面出现会覆盖前面，这里确保唯一
    # 如果同一天既有B又有S，保留B（更积极）
    conflict = (df["bs_signal"] == "B") & (df["bs_signal"].shift(-1) == "S")
    # 实际上上面赋值顺序可能导致覆盖，所以检查并清理
    # 如果同一天两者都为True，最后赋值的是S，但我们希望B优先，所以重新处理
    buy_mask = buy_cond
    sell_mask = sell_cond
    df["bs_signal"] = ""
    df["bs_type"] = ""
    df.loc[buy_mask, "bs_signal"] = "B"
    df.loc[buy_mask, "bs_type"] = "B"
    df.loc[sell_mask & ~buy_mask, "bs_signal"] = "S"
    df.loc[sell_mask & ~buy_mask, "bs_type"] = "S"

    return df

# ------------------- 回测函数 -------------------
def backtest_signal(df, signal_col):
    """
    根据信号列回测，信号值为 1 表示持有，0 表示空仓（或反之）
    但 BS 信号是事件，所以这里先转换成持仓状态
    """
    # 将B/S事件转为目标仓位：B -> 1, S -> 0, 其他保持
    df["target_pos"] = 0.0
    df.loc[df[signal_col] == "B", "target_pos"] = 1.0
    df.loc[df[signal_col] == "S", "target_pos"] = 0.0
    # 向前填充非0状态？如果target_pos是0且没有信号，可能来自之前持有后的S，之后应该保持0。所以用ffill。
    df["target_pos"] = df["target_pos"].ffill().fillna(0)
    # 实际持仓：次日执行，所以shift(1)
    df["position"] = df["target_pos"].shift(1).fillna(0)

    # 日收益率（收盘价变化）
    df["daily_return"] = df["close"].pct_change().fillna(0)
    df["strategy_return"] = df["position"] * df["daily_return"]
    df["strategy_nav"] = (1 + df["strategy_return"]).cumprod()

    return df

# ------------------- 主流程 -------------------
if st.button("运行分析"):
    try:
        df = load_etf_data(
            code,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d")
        )

        if df.empty:
            st.error("未获取到数据，请检查代码或日期")
            st.stop()

        # 计算指标
        df = add_technical_indicators(df, ma_short, ma_long, rsi_period, vol_ma_period)

        # 生成原有均线回撤信号
        df["signal_ma"] = "观望"
        buy_cond_ma = (
            (df["close"] > df[f"MA{ma_short}"]) &
            (df[f"MA{ma_short}"] > df[f"MA{ma_long}"]) &
            (df["drawdown"] > -dd_pct)
        )
        sell_cond_ma = (df["close"] < df[f"MA{ma_short}"]) | (df["drawdown"] <= -dd_pct)
        df.loc[buy_cond_ma, "signal_ma"] = "买入/持有"
        df.loc[sell_cond_ma, "signal_ma"] = "卖出/观望"

        # 金叉死叉覆盖
        df["MA_short_prev"] = df[f"MA{ma_short}"].shift(1)
        df["MA_long_prev"] = df[f"MA{ma_long}"].shift(1)
        df.loc[
            (df["MA_short_prev"] <= df["MA_long_prev"]) &
            (df[f"MA{ma_short}"] > df[f"MA{ma_long}"]),
            "signal_ma"
        ] = "金叉买入"
        df.loc[
            (df["MA_short_prev"] >= df["MA_long_prev"]) &
            (df[f"MA{ma_short}"] < df[f"MA{ma_long}"]),
            "signal_ma"
        ] = "死叉卖出"

        # 生成 BS 信号
        df = generate_bs_signals(df, ma_short, dd_pct, vol_ma_period)

        # 回测原有均线策略
        df["target_position_ma"] = 0
        df.loc[df["signal_ma"].isin(["买入/持有", "金叉买入"]), "target_position_ma"] = 1
        df["position_ma"] = df["target_position_ma"].shift(1).fillna(0)
        df["daily_return"] = df["close"].pct_change().fillna(0)
        df["ma_strategy_return"] = df["position_ma"] * df["daily_return"]
        df["ma_strategy_nav"] = (1 + df["ma_strategy_return"]).cumprod()

        # 回测 BS 策略
        df = backtest_signal(df, "bs_signal")

        # 基准净值
        df["benchmark_nav"] = (1 + df["daily_return"]).cumprod()

        # ------------------- 核心指标展示 -------------------
        latest = df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新收盘价", f"{latest['close']:.3f}")
        col2.metric(f"{ma_short}日均线", f"{latest[f'MA{ma_short}']:.3f}")
        col3.metric(f"{ma_long}日均线", f"{latest[f'MA{ma_long}']:.3f}")
        col4.metric("当前回撤", f"{latest['drawdown']:.2f}%")

        st.success(f"最新均线信号：{latest['signal_ma']}    最新BS信号：{latest['bs_signal']}")

        # ------------------- 回测统计（均线策略） -------------------
        st.subheader("均线回撤策略回测")
        total_return_ma = (df["ma_strategy_nav"].iloc[-1] - 1) * 100
        benchmark_return = (df["benchmark_nav"].iloc[-1] - 1) * 100
        days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        annual_return_ma = (df["ma_strategy_nav"].iloc[-1] ** (365 / days) - 1) * 100 if days > 0 else 0
        roll_max_ma = df["ma_strategy_nav"].cummax()
        drawdown_ma = (df["ma_strategy_nav"] - roll_max_ma) / roll_max_ma * 100
        max_drawdown_ma = drawdown_ma.min()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("策略总收益", f"{total_return_ma:.2f}%")
        col2.metric("基准总收益", f"{benchmark_return:.2f}%")
        col3.metric("策略年化收益", f"{annual_return_ma:.2f}%")
        col4.metric("策略最大回撤", f"{max_drawdown_ma:.2f}%")

        # ------------------- 回测统计（BS策略） -------------------
        st.subheader("BS指标策略回测")
        total_return_bs = (df["strategy_nav"].iloc[-1] - 1) * 100
        annual_return_bs = (df["strategy_nav"].iloc[-1] ** (365 / days) - 1) * 100 if days > 0 else 0
        roll_max_bs = df["strategy_nav"].cummax()
        drawdown_bs = (df["strategy_nav"] - roll_max_bs) / roll_max_bs * 100
        max_drawdown_bs = drawdown_bs.min()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BS策略总收益", f"{total_return_bs:.2f}%")
        col2.metric("BS策略年化收益", f"{annual_return_bs:.2f}%")
        col3.metric("BS策略最大回撤", f"{max_drawdown_bs:.2f}%")
        col4.metric("交易次数（B/S变化）", int((df["bs_signal"] != "").sum()))

        # ------------------- 图表：K线 + 均线 + BS标记 + 成交量 -------------------
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=("K线与均线 + BS信号", "成交量")
        )

        # K线
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="K线",
                increasing_line_color="red",
                decreasing_line_color="green"
            ),
            row=1, col=1
        )

        # 均线
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[f"MA{ma_short}"],
                name=f"{ma_short}日均线",
                line=dict(color="blue", width=1.2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[f"MA{ma_long}"],
                name=f"{ma_long}日均线",
                line=dict(color="orange", width=1.2)
            ),
            row=1, col=1
        )

        # BS点标记
        # B点：价格下方，红色向上箭头
        buy_points = df[df["bs_signal"] == "B"]
        if not buy_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_points["date"],
                    y=buy_points["low"] * 0.98,  # 稍微低于最低价
                    mode="markers+text",
                    name="B点买入",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="red",
                        line=dict(width=1, color="darkred")
                    ),
                    text=["B"] * len(buy_points),
                    textposition="bottom center",
                    textfont=dict(color="red", size=10)
                ),
                row=1, col=1
            )

        # S点：价格上方，绿色向下箭头
        sell_points = df[df["bs_signal"] == "S"]
        if not sell_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_points["date"],
                    y=sell_points["high"] * 1.02,  # 稍微高于最高价
                    mode="markers+text",
                    name="S点卖出",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="green",
                        line=dict(width=1, color="darkgreen")
                    ),
                    text=["S"] * len(sell_points),
                    textposition="top center",
                    textfont=dict(color="green", size=10)
                ),
                row=1, col=1
            )

        # 成交量
        colors = np.where(df["close"] >= df["open"], "red", "green")
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="成交量",
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        # 成交量均线
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[f"VOL_MA{vol_ma_period}"],
                name=f"{vol_ma_period}日量均",
                line=dict(color="purple", width=1)
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f"{code} K线、均线、BS信号与成交量",
            xaxis_title="日期",
            yaxis_title="价格",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # ------------------- 净值曲线对比 -------------------
        nav_fig = go.Figure()
        nav_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["ma_strategy_nav"],
            name="均线策略净值",
            line=dict(color="red")
        ))
        nav_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["strategy_nav"],
            name="BS策略净值",
            line=dict(color="blue")
        ))
        nav_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["benchmark_nav"],
            name="基准净值",
            line=dict(color="gray", dash="dash")
        ))
        nav_fig.update_layout(
            title="策略净值对比",
            xaxis_title="日期",
            yaxis_title="净值",
            hovermode="x unified"
        )
        st.plotly_chart(nav_fig, use_container_width=True)

        # ------------------- 最近数据表格 -------------------
        st.subheader("最近数据")
        st.dataframe(
            df[["date", "open", "high", "low", "close", "volume",
                f"MA{ma_short}", f"MA{ma_long}", "DIF", "DEA", "RSI",
                "drawdown", "signal_ma", "bs_signal",
                "ma_strategy_nav", "strategy_nav"]].tail(60),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"获取数据或计算失败：{e}")
