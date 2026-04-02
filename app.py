# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
import AMD_Tools4 as amd

app = Flask(__name__)

@app.route("/get_temp", methods=["POST"])
def get_climate_data():
    try:
        d = request.get_json()
        lat, lon = map(float, (d["lat"], d["lon"]))
        threshold = float(d["threshold"])
        gdd1_target = float(d["gdd1"])
        hosei = float(d.get("hosei", 0.0))
        
        # 日付フォーマットを強固な pandas.Timestamp に統一！
        ct1_start = pd.to_datetime(d["ct1_start"])
        ct1_end   = pd.to_datetime(d["ct1_end"])
        
        today = pd.Timestamp(datetime.utcnow().date())
        yesterday = today - pd.Timedelta(days=1)
        forecast_end = today + pd.Timedelta(days=26)
        
        # 基準年の設定
        current_year = today.year if today.month >= 4 else today.year - 1
        start_year_for_avg = current_year - 3
        target_year = ct1_start.year if ct1_start.month >= 4 else ct1_start.year - 1

        # 1. 平年値の計算
        all_years_data = []
        for year in range(start_year_for_avg, start_year_for_avg + 3):
            start_str = f"{year}-04-01"
            end_str = f"{year+1}-03-31"
            try:
                temp, tim, *_ = amd.GetMetData("TMP_mea", [start_str, end_str], [lat, lat, lon, lon])
                df = pd.DataFrame({
                    "datetime": pd.to_datetime(tim),
                    "tave": temp[:, 0, 0]
                })
                df["month_day"] = df["datetime"].dt.strftime("%m-%d")
                all_years_data.append(df[["month_day", "tave"]])
            except Exception:
                pass

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

        # 2. 全ての取得可能な「実測値」＋「26日後までの予報値」をかき集める
        df_available_list = []
        
        # 昨年度データ
        try:
            py_start = f"{current_year-1}-04-01"
            py_end   = f"{current_year}-03-31"
            t_py, tim_py, *_ = amd.GetMetData("TMP_mea", [py_start, py_end], [lat, lat, lon, lon])
            p_py, *_       = amd.GetMetData("APCPRA", [py_start, py_end], [lat, lat, lon, lon])
            df_available_list.append(pd.DataFrame({
                "date": pd.to_datetime(tim_py).dt.normalize(),
                "tave_real": t_py[:, 0, 0],
                "prcp_real": p_py[:, 0, 0]
            }))
        except Exception:
            pass

        # 今年度データ（実測＋予報26日分）
        try:
            cy_start = f"{current_year}-04-01"
            cy_end   = f"{current_year+1}-03-31"
            t_cy, tim_cy, *_ = amd.GetMetData("TMP_mea", [cy_start, cy_end], [lat, lat, lon, lon])
            p_cy, *_       = amd.GetMetData("APCPRA", [cy_start, cy_end], [lat, lat, lon, lon])
            df_available_list.append(pd.DataFrame({
                "date": pd.to_datetime(tim_cy).dt.normalize(),
                "tave_real": t_cy[:, 0, 0],
                "prcp_real": p_cy[:, 0, 0]
            }))
        except Exception:
            if today.month == 4 and today.day <= 7:
                for offset in [-1, 0, 1, 2, 3]:
                    try:
                        start_d = (today + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                        f_end_str = forecast_end.strftime("%Y-%m-%d")
                        if pd.to_datetime(start_d) > forecast_end: break
                        t_f, tim_f, *_ = amd.GetMetData("TMP_mea", [start_d, f_end_str], [lat, lat, lon, lon])
                        p_f, *_ = amd.GetMetData("APCPRA", [start_d, f_end_str], [lat, lat, lon, lon])
                        df_available_list.append(pd.DataFrame({
                            "date": pd.to_datetime(tim_f).dt.normalize(),
                            "tave_real": t_f[:, 0, 0],
                            "prcp_real": p_f[:, 0, 0]
                        }))
                        break
                    except Exception:
                        continue

        if df_available_list:
            df_available = pd.concat(df_available_list).drop_duplicates(subset=["date"], keep="last")
        else:
            df_available = pd.DataFrame(columns=["date", "tave_real", "prcp_real"])

        # ====================================================================
        # 3. 指定された年度の「完璧に織り交ぜられた」365日予測タイムラインを作成
        # ====================================================================
        start_this = pd.to_datetime(f"{target_year}-04-01")
        end_this = pd.to_datetime(f"{target_year + 1}-03-31")
        df_this = pd.DataFrame({"date": pd.date_range(start=start_this, end=end_this)})
        
        def assign_tag(d):
            if d <= yesterday: return "past"
            elif d <= forecast_end: return "forecast"
            else: return "normal"
            
        df_this["tag"] = df_this["date"].apply(assign_tag)
        df_this["month_day"] = df_this["date"].dt.strftime("%m-%d")

        # 平年値を結合
        if not df_avg.empty:
            df_this = df_this.merge(df_avg[["month_day", "tave_avg"]], on="month_day", how="left")
        else:
            df_this["tave_avg"] = 10.0

        # 実測・予報データを結合
        if not df_available.empty:
            df_this = df_this.merge(df_available, on="date", how="left")
        else:
            df_this["tave_real"] = np.nan
            df_this["prcp_real"] = np.nan

        # 織り交ぜ処理
        df_this["tave_this"] = df_this["tave_real"]
        mask_normal = df_this["tag"] == "normal"
        df_this.loc[mask_normal, "tave_this"] = np.nan 
        df_this["tave_this"] = df_this["tave_this"].fillna(df_this["tave_avg"]).round(1)
        df_this["prcp_this"] = df_this["prcp_real"].fillna(0.0).round(1)

        df_this.drop(columns=["month_day", "tave_avg", "tave_real", "prcp_real"], inplace=True)
        # ====================================================================

        # 4. 上部グラフ用の9日間予報を抽出
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

        # 5. 積算範囲 (CT1) の計算【強固な日付比較】
        mask_ct = (df_this["date"] >= ct1_start) & (df_this["date"] <= ct1_end)
        df_ct1 = df_this.loc[mask_ct].copy().reset_index(drop=True)

        if df_ct1.empty:
            closest_dict, corrected_dict, hist_dict = {}, {}, {}
        else:
            df_ct1["daily_ct"] = (df_ct1["tave_this"] - threshold).clip(lower=0).round(1)          
            df_ct1["cum_ct"] = df_ct1["daily_ct"].cumsum().round(1)
            df_ct1["daily_pr"] = df_ct1["prcp_this"].round(1)
            df_ct1["cum_pr"] = df_ct1["daily_pr"].cumsum().round(1)

            df_ct1["abs_diff"] = (df_ct1["
