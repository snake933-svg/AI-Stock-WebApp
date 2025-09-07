import streamlit as st
import requests

st.set_page_config(page_title="台股資料除錯", layout="wide")
st.title("🕵️‍♂️ 台股資料來源除錯模式")

st.warning("此為除錯專用版本。請將下方顯示的所有文字內容，完整複製並回傳給我，謝謝您的幫助！")

# --- 顯示原始資料內容 ---
def show_raw_data():
    try:
        st.info("正在嘗試下載 `上市公司` 列表...")
        url_l = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"
        response_l = requests.get(url_l, verify=False)
        response_l.raise_for_status()
        st.text_area("上市公司列表 (L.csv) 原始文字內容 (前1000字元)", response_l.text[:1000], height=250)

        st.info("正在嘗試下載 `上櫃公司` 列表...")
        url_o = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"
        response_o = requests.get(url_o, verify=False)
        response_o.raise_for_status()
        st.text_area("上櫃公司列表 (O.csv) 原始文字內容 (前1000字元)", response_o.text[:1000], height=250)
        
    except Exception as e:
        st.error(f"在除錯過程中發生錯誤：{e}")

show_raw_data()