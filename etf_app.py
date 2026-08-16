import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="ETF均线回撤策略", layout="wide")
st.title("ETF 均线回撤策略助手")

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

        # ------------------- 计算指标 -------------------
        df[f"MA{ma_short}"] = df["close"].rolling(ma_short).mean()
        df[f"MA{ma_long}"] = df["close"].rolling(ma_long).mean()

        # 当前回撤
        df["rolling_high"] = df["close"].cummax()
        df["drawdown"] = (df["close"] - df["rolling_high"]) / df["rolling_high"] * 100

        # ------------------- 信号生成 -------------------
        df["signal"] = "观望"
        buy_cond = (
            (df["close"] > df[f"MA{ma_short}"]) &
            (df[f"MA{ma_short}"] > df[f"MA{ma_long}"]) &
            (df["drawdown"] > -dd_pct)
        )
        sell_cond = (df["close"] < df[f"MA{ma_short}"]) | (df["drawdown"] <= -dd_pct)

        df.loc[buy_cond, "signal"] = "买入/持有"
        df.loc[sell_cond, "signal"] = "卖出/观望"

        # 金叉死叉覆盖
        df["MA_short_prev"] = df[f"MA{ma_short}"].shift(1)
        df["MA_long_prev"] = df[f"MA{ma_long}"].shift(1)

        df.loc[
            (df["MA_short_prev"] <= df["MA_long_prev"]) &
            (df[f"MA{ma_short}"] > df[f"MA{ma_long}"]),
            "signal"
        ] = "金叉买入"

        df.loc[
            (df["MA_short_prev"] >= df["MA_long_prev"]) &
            (df[f"MA{ma_short}"] < df[f"MA{ma_long}"]),
            "signal"
        ] = "死叉卖出"

        # ------------------- 策略回测 -------------------
        # 持仓状态：买入/持有、金叉买入 -> 1，其他 -> 0
        df["target_position"] = 0
        df.loc[df["signal"].isin(["买入/持有", "金叉买入"]), "target_position"] = 1

        # 使用次日开盘价执行，避免未来函数
        df["position"] = df["target_position"].shift(1).fillna(0)

        # 日收益率（开盘到收盘）
        df["daily_return"] = df["close"].pct_change().fillna(0)
        df["strategy_return"] = df["position"] * df["daily_return"]

        # 净值曲线
        df["strategy_nav"] = (1 + df["strategy_return"]).cumprod()
        df["benchmark_nav"] = (1 + df["daily_return"]).cumprod()

        # ------------------- 回测统计 -------------------
        total_return = (df["strategy_nav"].iloc[-1] - 1) * 100
        benchmark_return = (df["benchmark_nav"].iloc[-1] - 1) * 100

        # 年化收益率
        days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        if days > 0:
            annual_return = (df["strategy_nav"].iloc[-1] ** (365 / days) - 1) * 100
        else:
            annual_return = 0

        # 最大回撤
        roll_max = df["strategy_nav"].cummax()
        drawdown_series = (df["strategy_nav"] - roll_max) / roll_max * 100
        max_drawdown = drawdown_series.min()

        # 交易次数：target_position变化次数
        trade_count = (df["target_position"].diff().fillna(0) != 0).sum()

        # ------------------- 展示核心指标 -------------------
        latest = df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新收盘价", f"{latest['close']:.3f}")
        col2.metric(f"{ma_short}日均线", f"{latest[f'MA{ma_short}']:.3f}")
        col3.metric(f"{ma_long}日均线", f"{latest[f'MA{ma_long}']:.3f}")
        col4.metric("当前回撤", f"{latest['drawdown']:.2f}%")

        st.success(f"最新信号：{latest['signal']}")

        # 回测统计
        st.subheader("策略回测统计")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("策略总收益", f"{total_return:.2f}%")
        metric_col2.metric("基准总收益", f"{benchmark_return:.2f}%")
        metric_col3.metric("策略年化收益", f"{annual_return:.2f}%")
        metric_col4.metric("策略最大回撤", f"{max_drawdown:.2f}%")

        st.caption(f"交易次数：{trade_count} 次（信号变化导致仓位变化）")

        # ------------------- 图表：K线 + 均线 + 成交量 -------------------
        # 创建子图：主图K线+均线，副图成交量
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=("K线与均线", "成交量")
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

        fig.update_layout(
            title=f"{code} 价格、均线与成交量",
            xaxis_title="日期",
            yaxis_title="价格",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # 净值曲线
        nav_fig = go.Figure()
        nav_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["strategy_nav"],
            name="策略净值",
            line=dict(color="red")
        ))
        nav_fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["benchmark_nav"],
            name="基准净值",
            line=dict(color="gray", dash="dash")
        ))
        nav_fig.update_layout(
            title="策略 vs 基准 净值曲线",
            xaxis_title="日期",
            yaxis_title="净值",
            hovermode="x unified"
        )
        st.plotly_chart(nav_fig, use_container_width=True)

        # 最近数据
        st.subheader("最近数据")
        st.dataframe(
            df[["date", "open", "high", "low", "close", "volume",
                f"MA{ma_short}", f"MA{ma_long}", "drawdown", "signal",
                "position", "strategy_nav"]].tail(60),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"获取数据或计算失败：{e}")
