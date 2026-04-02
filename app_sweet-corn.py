# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import numpy as np
import math
import AMD_Tools4 as amd

app = Flask(__name__)

def fetch_met_data(element, start_str, end_str, lat, lon):
    try:
        val, tim, *_ = amd.GetMetData(element, [start_str, end_str], [lat, lat, lon, lon])
        s_dates = pd.to_datetime(pd.Series(list(tim)))
        df = pd.DataFrame({"date": s_dates.dt.normalize(), element: val[:, 0, 0]})
        return df
    except Exception:
        return pd.DataFrame(columns=["date", element])

@lru_cache(maxsize=128)
def get_cached_avg_data(year, lat, lon):
    df_t = fetch_met_data("TMP_mea", f"{year}-04-01", f"{year+1}-03-31", lat, lon)
    if df_t.empty: return None
    df_t["month_day"] = df_t["date"].dt.strftime("%m-%d")
    return df_t[["month_day", "TMP_mea"]].rename(columns={"TMP_mea": "tave_avg"})

@lru_cache(maxsize=64)
def get_cached_past_year(py_start, py_end, lat, lon):
    df_t = fetch_met_data("TMP_mea", py_start, py_end, lat, lon)
    df_p = fetch_met_data("APCPRA", py_start, py_end, lat, lon)
    if df_t.empty and df_p.empty: return None
    if not df_t.empty and not df_p.empty: df = df_t.merge(df_p, on="date", how="outer")
    else: df = df_t if not df_t.empty else df_p
    if "TMP_mea" not in df.columns: df["TMP_mea"] = np.nan
    if "APCPRA" not in df.columns: df["APCPRA"] = 0.0
    df.rename(columns={"TMP_mea": "tave_real", "APCPRA": "prcp_real"}, inplace=True)
    return df

@app.route("/get_temp", methods=["POST"])
def get_climate_data():
    try:
        d = request.get_json()
        lat, lon = map(float, (d["lat"], d["lon"]))
        threshold1, gdd1_target, hosei = float(d["threshold"]), float(d["gdd1"]), float(d.get("hosei", 0.0))
        ct1_start_str, ct1_end_str = d.get("ct1_start"), d.get("ct1_end")
        
        is_double = "ct2_start" in d and d["ct2_start"] != ""
        if is_double:
            threshold2, gdd2_target = float(d.get("threshold2", threshold1)), float(d.get("gdd2", 0))
            ct2_start_str, ct2_end_str = d.get("ct2_start"), d.get("ct2_end")

        today = pd.Timestamp(datetime.utcnow().date())
        yesterday, forecast_end = today - pd.Timedelta(days=1), today + pd.Timedelta(days=26)
        ct1_start_ts = pd.to_datetime(ct1_start_str) if ct1_start_str else today
        current_year = today.year if today.month >= 4 else today.year - 1
        target_year = ct1_start_ts.year if ct1_start_ts.month >= 4 else ct1_start_ts.year - 1

        all_years_data = [res for y in range(current_year - 3, current_year) if (res := get_cached_avg_data(y, lat, lon)) is not None]

        if all_years_data:
            df_avg = pd.concat(all_years_data).groupby("month_day", as_index=False).mean()
            df_avg["sort_key"] = pd.to_datetime("2000-" + df_avg["month_day"])
            df_avg = df_avg.sort_values("sort_key").reset_index(drop=True)
            idx_m = df_avg.index[df_avg["month_day"] == "04-01"]
            start_idx = idx_m[0] if len(idx_m) > 0 else 0
            df_avg = pd.concat([df_avg.iloc[start_idx:], df_avg.iloc[:start_idx]]).drop(columns=["sort_key"]).reset_index(drop=True)
            df_avg["tave_avg"] = df_avg["tave_avg"].round(1)
        else:
            df_avg = pd.DataFrame(columns=["month_day", "tave_avg"])

        df_available_list = []
        df_py = get_cached_past_year(f"{current_year-1}-04-01", f"{current_year}-03-31", lat, lon)
        if df_py is not None: df_available_list.append(df_py.copy())
        
        df_cy_t = fetch_met_data("TMP_mea", f"{current_year}-04-01", f"{current_year+1}-03-31", lat, lon)
        df_cy_p = fetch_met_data("APCPRA", f"{current_year}-04-01", f"{current_year+1}-03-31", lat, lon)
        if not df_cy_t.empty or not df_cy_p.empty:
            df_cy = df_cy_t.merge(df_cy_p, on="date", how="outer") if (not df_cy_t.empty and not df_cy_p.empty) else (df_cy_t if not df_cy_t.empty else df_cy_p)
            if "TMP_mea" not in df_cy.columns: df_cy["TMP_mea"] = np.nan
            if "APCPRA" not in df_cy.columns: df_cy["APCPRA"] = np.nan
            df_cy.rename(columns={"TMP_mea": "tave_real", "APCPRA": "prcp_real"}, inplace=True)
            df_available_list.append(df_cy)

        df_available = pd.concat(df_available_list).drop_duplicates(subset=["date"], keep="last") if df_available_list else pd.DataFrame(columns=["date", "tave_real", "prcp_real"])
        df_this = pd.DataFrame({"date": pd.date_range(start=f"{target_year}-04-01", end=f"{target_year+1}-03-31")})
        df_this["tag"] = df_this["date"].apply(lambda d: "past" if d <= yesterday else ("forecast" if d <= forecast_end else "normal"))
        df_this["month_day"] = df_this["date"].dt.strftime("%m-%d")
        df_this = df_this.merge(df_avg, on="month_day", how="left").merge(df_available, on="date", how="left")
        df_this["tave_this"] = np.where(df_this["tag"] == "normal", df_this["tave_avg"], df_this["tave_real"]).astype(float).fillna(df_this["tave_avg"]).round(1)
        df_this["prcp_this"] = np.where(df_this["tag"] == "normal", 0.0, df_this["prcp_real"]).astype(float).fillna(0.0).round(1)

        mask_f = (df_this["date"] >= (today - pd.Timedelta(days=1))) & (df_this["date"] <= (today + pd.Timedelta(days=7)))
        df_forecast = df_this.loc[mask_f].copy() if mask_f.any() else pd.DataFrame(columns=["date", "tave_this", "prcp_this"])

        def calc_acc(df_timeline, s_str, e_str, thresh, target):
            if not s_str or not e_str: return None, {}, {}, {}
            mask = (df_timeline["date"] >= pd.to_datetime(s_str)) & (df_timeline["date"] <= pd.to_datetime(e_str))
            df_ct = df_timeline.loc[mask].copy().reset_index(drop=True)
            if df_ct.empty: return None, {}, {}, {}
            df_ct["daily_ct"] = (df_ct["tave_this"] - thresh).clip(lower=0).round(1)
            df_ct["cum_ct"], df_ct["cum_pr"] = df_ct["daily_ct"].cumsum().round(1), df_ct["prcp_this"].cumsum().round(1)
            
            # ★ 未来の雨量は完全に消す
            mask_fut = df_ct["date"] > forecast_end
            df_ct.loc[mask_fut, ["daily_pr", "cum_pr"]] = np.nan
            
            row_cl = df_ct.loc[(df_ct["cum_ct"] - target).abs().idxmin()]
            cl_dict = {"date": row_cl["date"].strftime("%Y-%m-%d"), "cum_ct": round(row_cl["cum_ct"], 1)}
            corr_dict = {}
            if target > 0:
                df_ct["c_cum"] = df_ct["cum_ct"] + hosei
                row_co = df_ct.loc[(df_ct["c_cum"] - target).abs().idxmin()]
                corr_dict = {"date": row_co["date"].strftime("%Y-%m-%d"), "cum_ct": round(row_co["c_cum"], 1)}
            mask_h = df_ct["date"] <= yesterday
            h_dict = {"cum_ct": round(df_ct.loc[mask_h].iloc[-1]["cum_ct"], 1), "cum_pr": round(df_ct.loc[mask_h].iloc[-1]["cum_pr"], 1)} if mask_h.any() else {}
            return df_ct, cl_dict, corr_dict, h_dict

        df_ct1, cl1, co1, hi1 = calc_acc(df_this, ct1_start_str, ct1_end_str, threshold1, gdd1_target)
        res = {"average": df_avg.to_dict(orient="records"), "this_year": df_this.to_dict(orient="records"), "forecast": df_forecast.to_dict(orient="records"),
               "ct1": df_ct1.to_dict(orient="records") if df_ct1 is not None else [], "gdd1_target": cl1, "gdd1_target_corr": co1, "ct1_until_yesterday": hi1}
        if is_double:
            df_ct2, cl2, _, hi2 = calc_acc(df_this, ct2_start_str, ct2_end_str, threshold2, gdd2_target)
            res.update({"ct2": df_ct2.to_dict(orient="records") if df_ct2 is not None else [], "gdd2_target": cl2, "ct2_until_yesterday": hi2})

        def clean(obj):
            if isinstance(obj, list): return [clean(x) for x in obj]
            if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
            return "" if pd.isna(obj) else (obj.strftime("%Y-%m-%d") if isinstance(obj, pd.Timestamp) else obj)
        return jsonify(clean(res))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
