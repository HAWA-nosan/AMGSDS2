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
# ★ 爆速化＆堅牢化の要（日付変換のエラーを完全に撲滅！）
# ====================================================================
def fetch_met_data(element, start_str, end_str, lat, lon):
    try:
        val, tim, *_ = amd.GetMetData(element, [start_str, end_str], [lat, lat, lon, lon])
        s_dates = pd.to_datetime(pd.Series(list(tim)))
        df = pd.DataFrame({
            "date": s_dates.dt.normalize(),
            element: val[:, 0, 0]
        })
        return df
    except Exception as e:
        print(f"API Error ({element}): {e}")
        return pd.DataFrame(columns=["date", element])

@lru_cache(maxsize=128)
def get_cached_avg_data(year, lat, lon):
    start_str = f"{year}-04-01"
    end_str = f"{year+1}-03-31"
    df_t = fetch_met_data("TMP_mea", start_str, end_str, lat, lon)
    if df_t.empty: return None
    df_t["month_day"] = df_t["date"].dt.strftime("%m-%d")
    return df_t[["month_day", "TMP_mea"]].rename(columns={"TMP_mea": "tave_avg"})

@lru_cache(maxsize=64)
def get_cached_past_year(py_start, py_end, lat, lon):
    df_t = fetch_met_data("TMP_mea", py_start, py_end, lat, lon)
    df_p = fetch_met_data("APCPRA", py_start, py_end, lat, lon)
    if df_t.empty and df_p.empty: return None
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
        hosei = float(d.get("hosei", 0.0))
        ct1_start_str = d.get("ct1_start")
        ct1_end_str = d.get("ct1_end")
        
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
            if res is not None: all_years_data.append(res.copy())

        if all_years_data:
            df_concat = pd.concat(all_years_data)
            df_avg = df_concat.groupby("month_day", as_index=False).mean()
            df_avg["sort_key"] = pd.to_datetime("2000-" + df_avg["month_day"])
            df_avg = df_avg.sort_values("sort_key").reset_index(drop=True)
            start_idx = df_avg[df_avg["month_day"] == "04-01"].index[0]
            df_avg = pd.concat([df_avg.iloc[start_idx:], df_avg.iloc[:start_idx]]).drop(columns=["sort_key"]).reset_index(drop=True)
            df_avg["tave_avg"] = df_avg["tave_avg"].round(1)
        else:
            df_avg = pd.DataFrame(columns=["month_day", "tave_avg"])

        # 2. 実測値＋予報値の収集
        df_available_list = []
        py_start = f"{current_year-1}-04-01"
        py_end   = f"{current_year}-03-31"
        df_py = get_cached_past_year(py_start, py_end, lat, lon)
        if df_py is not None: df_available_list.append(df_py.copy())

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
            if "APCPRA" not in df_cy.columns: df_cy["APCPRA"] = np.nan
            df_cy.rename(columns={"TMP_mea": "tave_real", "APCPRA": "prcp_real"}, inplace=True)
            df_available_list.append(df_cy)
        else:
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
                        if "APCPRA" not in df_f.columns: df_f["APCPRA"] = np.nan
                        df_f.rename(columns={"TMP_mea": "tave_real", "APCPRA": "prcp_real"}, inplace=True)
                        df_available_list.append(df_f)
                        break

        if df_available_list:
            df_available = pd.concat(df_available_list).drop_duplicates(subset=["date"], keep="last")
        else:
            df_available = pd.DataFrame(columns=["date", "tave_real", "prcp_real"])

        # 3. タイムラインの作成
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

        mask_normal = df_this["tag"] == "normal"
        
        df_this["tave_this"] = df_this["tave_real"]
        df_this.loc[mask_normal, "tave_this"] = np.nan 
        df_this["tave_this"] = df_this["tave_this"].fillna(df_this["tave_avg"]).round(1)
        
        df_this["prcp_this"] = df_this["prcp_real"]
        df_this.loc[mask_normal, "prcp_this"] = 0.0 
        df_this["prcp_this"] = df_this["prcp_this"].fillna(0.0).round(1)
        
        df_this.drop(columns=["month_day", "tave_avg", "tave_real", "prcp_real"], inplace=True)

        # 4. グラフ用9日間予報抽出
        f_start_date = today - pd.Timedelta(days=1)
        f_end_date   = today + pd.Timedelta(days=7)
        mask_f = (df_available["date"] >= f_start_date) & (df_available["date"] <= f_end_date) if not df_available.empty else pd.Series(False, index=df_this.index)
        
        if mask_f.any():
            df_forecast = df_available.loc[mask_f].copy().reset_index(drop=True)
            df_forecast.rename(columns={"tave_real": "tave_this", "prcp_real": "prcp_this"}, inplace=True)
            df_forecast["prcp_this"] = df_forecast["prcp_this"].fillna(0.0)
        else:
            df_forecast = pd.DataFrame(columns=["date", "tave_this", "prcp_this"])

        # 5. 積算計算
        def calc_accumulation(df_timeline, start_str, end_str, thresh, target):
            if not start_str or not end_str: return pd.DataFrame(), {}, {}
            s_date = pd.to_datetime(start_str)
            e_date = pd.to_datetime(end_str)
            mask = (df_timeline["date"] >= s_date) & (df_timeline["date"] <= e_date)
            df_ct = df_timeline.loc[mask].copy().reset_index(drop=True)
            
            if df_ct.empty: return pd.DataFrame(), {}, {}
                
            df_ct["daily_ct"] = (df_ct["tave_this"] - thresh).clip(lower=0).round(1)
            df_ct["cum_ct"] = df_ct["daily_ct"].cumsum().round(1)
            df_ct["daily_pr"] = df_ct["prcp_this"].round(1)
            df_ct["cum_pr"] = df_ct["daily_pr"].cumsum().round(1)
            
            # ★ 未来の雨量は確実に空欄にする
            mask_future = df_ct["date"] > forecast_end
            df_ct.loc[mask_future, "daily_pr"] = np.nan
            df_ct.loc[mask_future, "cum_pr"] = np.nan
            
            df_ct["abs_diff"] = (df_ct["cum_ct"] - target).abs()
            row_close = df_ct.loc[df_ct["abs_diff"].idxmin()]
            closest_dict = {
                "date": row_close["date"].strftime("%Y-%m-%d"),
                "cum_ct": round(row_close["cum_ct"], 1)
            }

            if target > 0:
                df_ct["corrected_cum_ct"] = df_ct["cum_ct"] + hosei
                df_ct["abs_diff_corr"] = (df_ct["corrected_cum_ct"] - target).abs()
                row_corr = df_ct.loc[df_ct["abs_diff_corr"].idxmin()]
                corrected_dict = {
                    "date": row_corr["date"].strftime("%Y-%m-%d"),
                    "cum_ct": round(row_corr["corrected_cum_ct"], 1)
                }
            else:
                corrected_dict = {}
                
            mask_hist = df_ct["date"] <= yesterday
            if mask_hist.any():
                row_hist = df_ct.loc[mask_hist].iloc[-1]
                hist_dict = {"cum_ct": round(row_hist["cum_ct"], 1), "cum_pr": round(row_hist["cum_pr"], 1)}
            else:
                hist_dict = {}
            return df_ct, closest_dict, corrected_dict, hist_dict

        df_ct1, closest_dict1, corrected_dict1, hist_dict1 = calc_accumulation(df_this, ct1_start_str, ct1_end_str, threshold1, gdd1_target)

        # 6. JSONクリーンアップ
        def replace_nan(d):
            if isinstance(d, list): return [replace_nan(x) for x in d]
            if isinstance(d, dict): return {k: replace_nan(v) for k, v in d.items()}
            if pd.isna(d): return "" 
            return d

        df_this["date"] = df_this["date"].dt.strftime("%Y-%m-%d")
        df_forecast["date"] = df_forecast["date"].dt.strftime("%Y-%m-%d")
        if not df_ct1.empty: df_ct1["date"] = df_ct1["date"].dt.strftime("%Y-%m-%d")

        res_dict = {
            "average": replace_nan(df_avg.to_dict(orient="records")),
            "this_year": replace_nan(df_this.to_dict(orient="records")),
            "forecast": replace_nan(df_forecast.to_dict(orient="records")),
            "ct1": replace_nan(df_ct1.to_dict(orient="records")),
            "gdd1_target": closest_dict1,
            "gdd1_target_corr": corrected_dict1,
            "ct1_until_yesterday": hist_dict1,
        }
        return jsonify(res_dict)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
