# 安裝必要套件
# !pip install streamlit yfinance scikit-learn pandas matplotlib --quiet

import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="手機專用台股 AI 選股", layout="wide")
st.title("📈 手機專用台股 AI 選股")

# 輸入股票代號
symbol = st.text_input("請輸入台股股票代號，例如 2330", "")

if symbol:
    symbol_full = symbol + ".TW"
    data = yf.download(symbol_full, period="2y", auto_adjust=True)
    if data.empty:
        st.error(f"找不到股票代號 {symbol_full}")
    else:
        # 技術指標
        data["Return"] = data["Close"].pct_change()
        data["MA5"] = data["Close"].rolling(5).mean()
        data["MA20"] = data["Close"].rolling(20).mean()
        data["RSI"] = 100 - (100 / (1 + (data["Return"].rolling(14).mean() / abs(data["Return"].rolling(14).mean()))))
        data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = data["EMA12"] - data["EMA26"]
        data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD_Hist"] = data["MACD"] - data["Signal"]

        data = data.dropna()
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data = data[:-1]

        # AI 模型
        X = data[["MA5", "MA20", "RSI"]]
        y = data["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        prob_up = model.predict_proba(X.tail(1))[0][1]

        # 最新收盤價與漲跌幅
        last_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2]
        change_pct = (last_close - prev_close) / prev_close * 100

        # 顯示數字結果
        st.subheader(f"{symbol_full} 數字結果")
        st.write(f"💰 最新收盤價：{last_close:.2f}")
        st.write(f"📊 今日漲跌幅：{change_pct:.2f}%")
        st.write(f"🎯 模型準確率：{accuracy:.2%}")
        st.write(f"🔮 預測明日上漲機率：{prob_up:.2%}")

        # 圖表
        if st.checkbox("顯示圖表"):
            st.subheader("股價圖表")
            fig, ax = plt.subplots(3,1,figsize=(10,8))
            # 主圖
            ax[0].plot(data.index, data["Close"], label="收盤價")
            ax[0].plot(data.index, data["MA5"], label="MA5")
            ax[0].plot(data.index, data["MA20"], label="MA20")
            ax[0].fill_between(data.index, data["MA20"]+2*data["Close"].rolling(20).std(),
                                data["MA20"]-2*data["Close"].rolling(20).std(), color="gray", alpha=0.2, label="布林通道")
            ax[0].legend()
            # 成交量
            colors = ["red" if c>=0 else "green" for c in data["Return"]]
            ax[1].bar(data.index, data["Volume"], color=colors)
            ax[1].set_ylabel("成交量")
            # RSI + MACD
            ax[2].plot(data.index, data["RSI"], label="RSI", color="purple")
            ax[2].plot(data.index, data["MACD"], label="MACD", color="blue")
            ax[2].plot(data.index, data["Signal"], label="Signal", color="orange")
            ax[2].bar(data.index, data["MACD_Hist"], color="gray", alpha=0.5)
            ax[2].legend()
            st.pyplot(fig)