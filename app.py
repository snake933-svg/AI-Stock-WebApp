import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(page_title="台股資料除錯", layout="wide")
st.title("🕵️‍♂️ 欄位名稱除錯模式")

st.warning("此為除錯專用版本。請將下方顯示的所有文字內容，完整複製並回傳給我，謝謝您的幫助！")

# --- 顯示原始資料內容 ---
def show_column_names():
    try:
        st.info("正在嘗試下載 `上市公司` 列表...")
        url_l = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"
        response_l = requests.get(url_l, verify=False)
        response_l.raise_for_status()
        df_l = pd.read_csv(io.StringIO(response_l.text), encoding='utf-8-sig')
        st.subheader("「上市公司」列表的實際欄位名稱是:")
        st.write(list(df_l.columns))
        st.text_area("上市公司列表原始文字內容 (前500字元)", response_l.text[:500], height=150)

        st.info("正在嘗試下載 `上櫃公司` 列表...")
        url_o = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"
        response_o = requests.get(url_o, verify=False)
        response_o.raise_for_status()
        df_o = pd.read_csv(io.StringIO(response_o.text), encoding='utf-8-sig')
        st.subheader("「上櫃公司」列表的實際欄位名稱是:")
        st.write(list(df_o.columns))
        st.text_area("上櫃公司列表原始文字內容 (前500字元)", response_o.text[:500], height=150)
        
    except Exception as e:
        st.error(f"在除錯過程中發生錯誤：{e}")

show_column_names()
