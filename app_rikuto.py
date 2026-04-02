# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import numpy as np
import math
import AMD_Tools4 as amd

app = Flask(__name__)

# ====================================================================
# ★ 爆速化＆堅牢化の要
# ====================================================================
def fetch_met_data(element, start_str, end_str, lat, lon):
    """気温や降水量を個別に取得し、エラー時は空のDataFrameを返す安全な関数"""
    try:
        val, tim, *_ = amd.GetMetData(element, [start_str, end_str], [lat, lat, lon, lon])
        return pd.DataFrame({
            "date": pd.to_datetime(tim).dt.normalize(),
            element: val[:, 0, 0]
        })
    except Exception:
        return pd.DataFrame(columns=["date", element])

@lru_cache(maxsize=128)
def get_cached_avg_data(year, lat, lon):
    start_str = f"{year}-04-01"
    end_str = f"{year+1}-03-31"
    df_t = fetch_met_data("TMP_mea", start_str, end_str, lat, lon)
    if df_t.empty:
        return None
    df_t["month_day"] = df_t["date"].dt.strftime("%m-%d")
    return df_t[["month_day", "TMP_mea"]].rename(columns={"TMP_mea": "tave"})

@lru_cache(maxsize=64)
def get_cached_past_year(py_start, py_end, lat, lon):
    """過去のデータを気温・降水量それぞれ独立して取得する"""
    df_t = fetch_met_data("TMP_mea", py_start, py_end, lat, lon)
    df_p = fetch_met_data("APCPRA", py_start, py_end, lat, lon)
    
    if df_t.empty and df_p.empty:
        return None
        
    # 片方だけ成功した場合もガッチャンコして返す
    if not df_t.empty and not df_p.empty:
        df = df_t.merge(df_p, on="date", how="outer")
    else:
        df = df_t if not df_t.empty else df_p
        
    if "TMP_mea" not in df.columns: df["TMP_mea"] = np.nan
    if "APCPRA" not in df.columns: df["APCPRA"] = 0.0
    
    df.rename(columns={"TMP_mea": "tave_real", "APCPRA": "prcp_real"}, inplace=True)
    return df

@app.route("/get_temp", methods=["POST"])
def get_climate_data():
    try:
        d = request.get_json()
        lat, lon = map(float, (d["lat"], d["lon"]))
        
        threshold1 = float(d["threshold"])
        gdd1_target = float(d["gdd1"])
        ct1_start_str = d.get("ct1_start")
        ct1_end_str = d.get("ct1_end")
        
        threshold2 = float(d.get("threshold2", threshold1))
        gdd2_target = float(d.get("gdd2", 0))
        ct2_start_str = d.get("ct2_start")
        ct2_end_str = d.get("ct2_end")
        
        today = pd.Timestamp(datetime.utcnow().date())
        yesterday = today - pd.Timedelta(days=1)
        forecast_end = today + pd.Timedelta(days=26)
        
        ct1_start_ts = pd.to_datetime(ct1_start_str) if ct1_start_str else today
        
        current_year = today.year if today.month >= 4 else today.year - 1
        start_year_for_avg = current_year - 3
        target_year = ct1_start_ts.year if ct1_start_ts.month >= 4 else ct1_start_ts.year - 1

        # 1. 平年値の計算
        all_years_data = []
        for year in range(start_year_for_avg, start_year_for_avg + 3):
            res = get_cached_avg_data(year, lat, lon)
            if res is not None:
                all_years_data.append(res.copy())

        if all_years_data:
            df_concat = pd.concat(all_years_data)
            df_avg = df_concat.groupby("month_day", as_index=False)["tave"].mean()
            df_avg.rename(columns={"tave": "tave_avg"}, inplace=True)
            df_avg["sort_key"] = pd.to_datetime("2000-" + df_avg["month_day"])
            df_avg = df_avg.sort_values("sort_key").reset_index(drop=True)
            start_idx = df_avg[df_avg["month_day"] == "04-01"].index[0]
            df_avg = pd.concat([df_avg.iloc[start_idx:], df_avg.iloc[:start_idx]]).drop(columns=["sort_key"]).reset_index(drop=True)
            df_avg["tave_avg"] = df_avg["tave_avg"].round(1)
        else:
            df_avg = pd.DataFrame(columns=["month_day", "tave_avg"])

        # 2. 実測値＋予報値の収集（分離取得で絶対に落とさない）
        df_available_list = []
        
        py_start = f"{current_year-1}-04-01"
        py_end   = f"{current_year}-03-31"
        df_py = get_cached_past_year(py_start, py_end, lat, lon)
        if df_py is not None:
            df_available_list.append(df_py.copy())

        # 今年度データ
        cy_start = f"{current_year}-04-01"
        cy_end   = f"{current_year+1}-03-31"
        df_cy_t = fetch_met_data("TMP_mea", cy_start, cy_end, lat, lon)
        df_cy_p = fetch_met_data("APCPRA", cy_start, cy_end, lat, lon)
        
        if not df_cy_t.empty or not df_cy_p.empty:
            if not df_cy_t.empty and not df_cy_p.empty:
                df_cy = df_cy_t.merge(df_cy_p, on="date", how="outer")
            else:
                df_cy = df_cy_t if not df_cy_t.empty else df_cy_p
                
            if "TMP_mea" not in df_cy.columns: df_cy["TMP_mea"] = np.nan
            if "APCPRA" not in df_cy.columns: df_cy["APCPRA"] = 0.0
            df_cy.rename(columns={"TMP_mea": "tave_real", "APCPRA": "prcp_real"}, inplace=True)
            df_available_list.append(df_cy)
        else:
            # 4月上旬バグ用のフォールバック
            if today.month == 4 and today.day <= 7:
                for offset in [-1, 0, 1, 2, 3]:
                    start_d = (today + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                    f_end_str = forecast_end.strftime("%Y-%m-%d")
                    if pd.to_datetime(start_d) > forecast_end: break
                    
                    df_f_t = fetch_met_data("TMP_mea", start_d, f_end_str, lat, lon)
                    df_f_p = fetch_met_data("APCPRA", start_d, f_end_str, lat, lon)
                    
                    if not df_f_t.empty or not df_f_p.empty:
                        if not df_f_t.empty and not df_f_p.empty:
                            df_f = df_f_t.merge(df_f_p, on="date", how="outer")
                        else:
                            df_f = df_f_t if not df_f_t.empty else df_f_p
                            
                        if "TMP_mea" not in df_f.columns: df_f["TMP_mea"] = np.nan
                        if "APCPRA" not in df_f.columns: df_f["APCPRA"] = 0.0
                        df_f.rename(columns={"TMP_mea": "tave_real", "APCPRA": "prcp_real"}, inplace=True)
                        df_available_list.append(df_f)
                        break

        if df_available_list:
            df_available = pd.concat(df_available_list).drop_duplicates(subset=["date"], keep="last")
        else:
            df_available = pd.DataFrame(columns=["date", "tave_real", "prcp_real"])

        # 3. 365日予測タイムライン（ハイブリッド）の作成
        start_this = pd.to_datetime(f"{target_year}-04-01")
        end_this = pd.to_datetime(f"{target_year + 1}-03-31")
        df_this = pd.DataFrame({"date": pd.date_range(start=start_this, end=end_this)})
        
        def assign_tag(d):
            if d <= yesterday: return "past"
            elif d <= forecast_end: return "forecast"
            else: return "normal"
            
        df_this["tag"] = df_this["date"].apply(assign_tag)
        df_this["month_day"] = df_this["date"].dt.strftime("%m-%d")

        if not df_avg.empty:
            df_this = df_this.merge(df_avg[["month_day", "tave_avg"]], on="month_day", how="left")
        else:
            df_this["tave_avg"] = 10.0

        if not df_available.empty:
            df_this = df_this.merge(df_available, on="date", how="left")
        else:
            df_this["tave_real"] = np.nan
            df_this["prcp_real"] = np.nan

        df_this["tave_this"] = df_this["tave_real"]
        mask_normal = df_this["tag"] == "normal"
        df_this.loc[mask_normal, "tave_this"] = np.nan 
        df_this["tave_this"] = df_this["tave_this"].fillna(df_this["tave_avg"]).round(1)
        df_this["prcp_this"] = df_this["prcp_real"].fillna(0.0).round(1)
        df_this.drop(columns=["month_day", "tave_avg", "tave_real", "prcp_real"], inplace=True)

        # 4. グラフ用9日間予報抽出
        f_start_date = today - pd.Timedelta(days=1)
        f_end_date   = today + pd.Timedelta(days=7)
        mask_f = (df_available["date"] >= f_start_date) & (df_available["date"] <= f_end_date) if not df_available.empty else pd.Series(False, index=df_this.index)
        
        if mask_f.any():
            df_forecast = df_available.loc[mask_f].copy().reset_index(drop=True)
            df_forecast.rename(columns={"tave_real": "tave_this", "prcp_real": "prcp_this"}, inplace=True)
        else:
            df_forecast = pd.DataFrame(columns=["date", "tave_this", "prcp_this"])

        if df_forecast.empty or len(df_forecast) < 9:
            dates_9 = pd.date_range(start=f_start_date, periods=9)
            df_fb = pd.DataFrame({"date": dates_9, "prcp_this": 0.0})
            df_fb["month_day"] = df_fb["date"].dt.strftime("%m-%d")
            if not df_avg.empty:
                df_fb = df_fb.merge(df_avg[["month_day", "tave_avg"]], on="month_day", how="left")
                df_fb["tave_this"] = df_fb["tave_avg"].fillna(10.0).round(1)
            else:
                df_fb["tave_this"] = 10.0
            
            if not df_forecast.empty:
                df_fb = df_fb.merge(df_forecast, on="date", how="left", suffixes=("", "_real"))
                df_fb["tave_this"] = df_fb["tave_this_real"].fillna(df_fb["tave_this"])
                df_fb["prcp_this"] = df_fb["prcp_this_real"].fillna(df_fb["prcp_this"])
                df_fb.drop(columns=["tave_this_real", "prcp_this_real"], inplace=True)
                
            df_forecast = df_fb.drop(columns=["month_day", "tave_avg"], errors='ignore')

        # ====================================================================
        # 5. 積算計算
        # ====================================================================
        def calc_accumulation(df_timeline, start_str, end_str, thresh, target):
            if not start_str or not end_str:
                return pd.DataFrame(), {}, {}
                
            s_date = pd.to_datetime(start_str)
            e_date = pd.to_datetime(end_str)
            
            mask = (df_timeline["date"] >= s_date) & (df_timeline["date"] <= e_date)
            df_ct = df_timeline.loc[mask].copy().reset_index(drop=True)
            
            if df_ct.empty:
                return pd.DataFrame(), {}, {}
                
            df_ct["daily_ct"] = (df_ct["tave_this"] - thresh).clip(lower=0).round(1)
            df_ct["cum_ct"] = df_ct["daily_ct"].cumsum().round(1)
            df_ct["daily_pr"] = df_ct["prcp_this"].round(1)
            df_ct["cum_pr"] = df_ct["daily_pr"].cumsum().round(1)
            
            df_ct["abs_diff"] = (df_ct["cum_ct"] - target).abs()
            row_close = df_ct.loc[df_ct["abs_diff"].idxmin()]
            closest_dict = {
                "date": row_close["date"].strftime("%Y-%m-%d"),
                "cum_ct": round(row_close["cum_ct"], 1)
            }
            
            mask_hist = df_ct["date"] <= yesterday
            if mask_hist.any():
                row_hist = df_ct.loc
