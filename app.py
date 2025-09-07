import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import requests
import io

st.set_page_config(page_title="台股 AI 分析", layout="wide")
st.title("📈 台股 AI 分析與預測")

# --- 資料載入與快取 ---
@st.cache_data(ttl=86400) # 快取資料一天
def load_stock_list():
    """載入並合併上市與上櫃公司列表"""
    try:
        url_l = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"
        url_o = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"
        
        response_l = requests.get(url_l, verify=False)
        response_l.raise_for_status()
        response_o = requests.get(url_o, verify=False)
        response_o.raise_for_status()

        df_l = pd.read_csv(io.StringIO(response_l.text), encoding='utf-8-sig', header=0)
        df_o = pd.read_csv(io.StringIO(response_o.text), encoding='utf-8-sig', header=0)
        
        df_l = df_l.iloc[:, [1, 3]]
        df_l.columns = ['code', 'name']
        df_o = df_o.iloc[:, [1, 3]]
        df_o.columns = ['code', 'name']

        df_l['type'] = '上市'
        df_o['type'] = '上櫃'
        
        stock_list = pd.concat([df_l, df_o])
        stock_list['code'] = stock_list['code'].astype(str).str.strip()
        stock_list = stock_list.drop_duplicates(subset='code', keep='first')
        return stock_list.set_index('code')
    except Exception as e:
        st.error(f"處理股票列表時發生錯誤，請重新整理頁面。錯誤訊息：{e}")
        return None

stock_list = load_stock_list()

if stock_list is not None:
    symbol = st.text_input("請輸入台股股票代號 (例如 2330, 8109)", "").strip()

    if symbol:
        if symbol in stock_list.index:
            stock_info = stock_list.loc[symbol]
            stock_name = stock_info['name']
            stock_type = stock_info['type']
            
            st.subheader(f"{symbol} {stock_name} ({stock_type})")

            suffix = ".TW" if stock_type == '上市' else ".TWO"
            symbol_full = symbol + suffix
            
            try:
                data = yf.download(symbol_full, period="2y", auto_adjust=True)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                    data = data.loc[:,~data.columns.duplicated()]

                if data.empty:
                    st.error(f"❌ 找不到 {symbol_full} 的股價資料。")
                else:
                    data["Return"] = data["Close"].pct_change()
                    data["MA5"] = data["Close"].rolling(5).mean()
                    data["MA20"] = data["Close"].rolling(20).mean()
                    data["UpperBB"] = data["MA20"] + 2 * data["Close"].rolling(20).std()
                    data["LowerBB"] = data["MA20"] - 2 * data["Close"].rolling(20).std()
                    
                    delta = data['Close'].diff(1)
                    gain = delta.mask(delta < 0, 0)
                    loss = -delta.mask(delta > 0, 0)
                    avg_gain = gain.ewm(com=13, adjust=False).mean()
                    avg_loss = loss.ewm(com=13, adjust=False).mean()
                    rs = avg_gain / avg_loss
                    data['RSI'] = 100 - (100 / (1 + rs))

                    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
                    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
                    data["MACD"] = data["EMA12"] - data["EMA26"]
                    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
                    data["MACD_Hist"] = data["MACD"] - data["Signal"]

                    data = data.dropna()
                    
                    if len(data) < 50:
                        st.error("有效資料量過少，無法建立可靠的 AI 模型。")
                    else:
                        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
                        data = data[:-1]

                        X = data[["MA5", "MA20", "RSI", "MACD"]]
                        y = data["Target"]

                        X.replace([np.inf, -np.inf], np.nan, inplace=True)
                        if X.isnull().values.any():
                            y = y[~X.isnull().any(axis=1)]
                            X = X.dropna()

                        if len(X) < 50:
                            st.error("有效資料量過少，無法建立可靠的 AI 模型。")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                            
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            
                            accuracy = accuracy_score(y_test, model.predict(X_test))
                            prob_up = model.predict_proba(X.tail(1))[0][1]

                            last_close = data["Close"].iloc[-1]
                            prev_close = data["Close"].iloc[-2]
                            change_pct = (last_close - prev_close) / prev_close * 100

                            st.subheader(f"數據總覽")
                            col1, col2 = st.columns(2)
                            col1.metric("最新收盤價", f"{last_close:.2f}", f"{change_pct:.2f}%")
                            col2.metric("AI 預測明日上漲機率", f"{prob_up:.2%}", f"模型準確率: {accuracy:.2%}")

                            if st.checkbox("顯示詳細技術圖表"):
                                st.subheader("股價圖表")
                                fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                                
                                ax[0].set_title(f"{symbol} {stock_name} 股價走勢", fontsize=16)
                                ax[0].plot(data.index, data["Close"], label="收盤價", linewidth=1.5)
                                ax[0].plot(data.index, data["MA5"], label="MA5", linewidth=1, linestyle='--')
                                ax[0].plot(data.index, data["MA20"], label="MA20", linewidth=1, linestyle='--')
                                ax[0].fill_between(data.index, data["UpperBB"], data["LowerBB"], color="gray", alpha=0.2, label="布林通道")
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
                                ax2_twin.plot(data.index, data["Signal"], label="Signal", color="orange", linewidth=1, linestyle='--')
                                ax2_twin.bar(data.index, data["MACD_Hist"], color="grey", alpha=0.3, label="MACD Hist")
                                ax2_twin.set_ylabel("MACD")
                                ax2_twin.legend(loc='upper right')

                                plt.xlabel("日期")
                                plt.tight_layout()
                                st.pyplot(fig)

            except Exception as e:
                st.error(f"抓取或分析股價時發生錯誤：{e}")
        else:
            st.warning("請輸入有效的台股上市或上櫃公司代號。")