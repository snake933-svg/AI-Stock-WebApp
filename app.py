# 安裝必要套件
# !pip install streamlit yfinance scikit-learn pandas matplotlib --quiet

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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

  try:
      data = yf.download(symbol_full, period="2y", auto_adjust=True)
      if data.empty:
          st.error(f"❌ 找不到股票代號 {symbol_full} 的資料，請確認代號是否正確。")
      else:
          # --- 技術指標計算 (包含錯誤修正) ---
          data["Return"] = data["Close"].pct_change()
          data["MA5"] = data["Close"].rolling(5).mean()
          data["MA20"] = data["Close"].rolling(20).mean()
          data["UpperBB"] = data["MA20"] + 2 * data["Close"].rolling(20).std()
          data["LowerBB"] = data["MA20"] - 2 * data["Close"].rolling(20).std()

          # --- 更穩健的 RSI 計算方法 ---
          delta = data['Close'].diff(1)
          gain = delta.mask(delta < 0, 0)
          loss = -delta.mask(delta > 0, 0)
          avg_gain = gain.ewm(com=13, adjust=False).mean()
          avg_loss = loss.ewm(com=13, adjust=False).mean()
          rs = avg_gain / avg_loss
          data['RSI'] = 100 - (100 / (1 + rs))

          # --- MACD 計算 ---
          data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
          data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
          data["MACD"] = data["EMA12"] - data["EMA26"]
          data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
          data["MACD_Hist"] = data["MACD"] - data["Signal"]

          # 移除因計算指標產生的空值 (NaN)
          data = data.dropna()

          if data.empty:
               st.error("資料不足，無法進行分析。請嘗試其他股票。")
          else:
              data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
              data = data[:-1]

              # --- AI 模型 ---
              X = data[["MA5", "MA20", "RSI", "MACD"]]
              y = data["Target"]

              # 再次確認 X 中沒有無限大的值
              X.replace([np.inf, -np.inf], np.nan, inplace=True)
              if X.isnull().values.any():
                  # 如果存在空值，則移除這些行
                  y = y[~X.isnull().any(axis=1)]
                  X = X.dropna()

              if X.empty:
                  st.error("資料清理後不足，無法訓練模型。請嘗試其他股票。")
              else:
                  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
shuffle=False, stratify=None)

                  model = RandomForestClassifier(n_estimators=100, random_state=42)
                  model.fit(X_train, y_train)

                  accuracy = accuracy_score(y_test, model.predict(X_test))
                  prob_up = model.predict_proba(X.tail(1))[0][1]

                  # --- 顯示結果 ---
                  last_close = data["Close"].iloc[-1]
                  prev_close = data["Close"].iloc[-2]
                  change_pct = (last_close - prev_close) / prev_close * 100

                  st.subheader(f"{symbol_full} 數據總覽")
                  col1, col2 = st.columns(2)
                  col1.metric("最新收盤價", f"{last_close:.2f}", f"{change_pct:.2f}%")
                  col2.metric("AI 預測明日上漲機率", f"{prob_up:.2%}", f"模型準確率: {accuracy:.2%}")

                  if st.checkbox("顯示詳細技術圖表"):
                      st.subheader("股價圖表")
                      fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

                      ax[0].set_title(f"{symbol_full} 股價走勢", fontsize=16)
                      ax[0].plot(data.index, data["Close"], label="收盤價", linewidth=1.5)
                      ax[0].plot(data.index, data["MA5"], label="MA5", linewidth=1, linestyle='--')
                      ax[0].plot(data.index, data["MA20"], label="MA20", linewidth=1, linestyle='--')
                      ax[0].fill_between(data.index, data["UpperBB"], data["LowerBB"], color="gray",
alpha=0.2, label="布林通道")
                      ax[0].legend()
                      ax[0].grid(True)

                      colors = ["#ff4d4d" if c >= 0 else "#4caf50" for c in data["Return"]]
                      ax[1].bar(data.index, data["Volume"], color=colors, width=1.0, alpha=0.6)
                      ax[1].set_ylabel("成交量")
                      ax[1].grid(True)

                      ax[2].plot(data.index, data["RSI"], label="RSI (14)", color="purple", linewidth=1)
                      ax[2].axhline(70, color="red", linestyle="--", linewidth=0.8)
                      ax[2].axhline(30, color="green", linestyle="--", linewidth=0.8)
                      ax[2].set_ylabel("RSI")
                      ax[2].legend(loc='upper left')

                      ax2_twin = ax[2].twinx()
                      ax2_twin.plot(data.index, data["MACD"], label="MACD", color="blue", linewidth=1)
                      ax2_twin.plot(data.index, data["Signal"], label="Signal", color="orange",
linewidth=1, linestyle='--')
                      ax2_twin.bar(data.index, data["MACD_Hist"], color="grey", alpha=0.3, label="MACD
Hist")
                      ax2_twin.set_ylabel("MACD")
                      ax2_twin.legend(loc='upper right')

                      plt.xlabel("日期")
                      plt.tight_layout()
                      st.pyplot(fig)

  except Exception as e:
      st.error(f"發生錯誤：{e}")
