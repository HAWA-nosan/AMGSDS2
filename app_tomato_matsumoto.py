# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta, date
from functools import lru_cache
import pandas as pd
import numpy as np
import traceback
import os

try:
    import AMD_Tools4 as amd
except ImportError:
    amd = None

app = Flask(__name__)

# エラー時にJSONを返す安全装置
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"status": "error", "message": "Server Crash", "trace": traceback.format_exc()}), 500

# GASからの「生存確認（スリープ防止）」に応答するルート
@app.route("/", methods=["GET"])
@app.route("/ping", methods=["GET"])
def ping():
    return "OK Server is awake!", 200

# 約1kmのメッシュ単位で同一視してキャッシュ（記憶）する防弾仕様
@lru_cache(maxsize=1024)
def _cached_fetch(var_name: str, start_date: str, end_date: str, mesh_code: str, center_lat: float, center_lon: float):
    # 通信失敗やトークン切れの際は明確にエラーを発生させ、シート破壊を防ぐ
    arr, tim, *_ = amd.GetMetData(var_name, [start_date, end_date], [center_lat, center_lat, center_lon, center_lon])
    values = arr[:, 0, 0]
    s_dates = pd.to_datetime(pd.Series(list(tim)))
    dates = s_dates.dt.normalize().dt.date.tolist()
    return dates, list(values)

def fetch_point_series_bulk(var_name: str, start_date: str, end_date: str, lat: float, lon: float):
    mesh_code = amd.lalo2mesh(lat, lon)
    center_lat, center_lon = amd.mesh2lalo(mesh_code)
    return _cached_fetch(var_name, start_date, end_date, mesh_code, center_lat, center_lon)

@app.route("/get_temp", methods=["POST"])
def get_climate_data():
    try:
        d = request.get_json()
        
        # 🚀【トークン金庫化①】GASから渡されたトークンをシステム変数に一時セット
        incoming_token = d.get("amd_token")
        if incoming_token:
            os.environ["AMD_TOKEN_JSON"] = incoming_token

        lat, lon = float(d["lat"]), float(d["lon"])
        threshold = float(d["threshold"])
        gdd1_target = float(d["gdd1"])
        hosei = float(d.get("hosei", 0.0))
        
        ct1_start = pd.to_datetime(d["ct1_start"]).date()  
        ct1_end   = pd.to_datetime(d["ct1_end"]).date()
        
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        forecast_end = today + timedelta(days=26)
        
        real_this_year = today.year if today.month >= 4 else today.year - 1

        # --- ① 過去3年平均気温を算出 ---
        all_years_data = []
        for year in range(real_this_year - 3, real_this_year):
            start, end = f"{year}-04-01", f"{year+1}-03-31"
            dates, tmean = fetch_point_series_bulk("TMP_mea", start, end, lat, lon)
            if dates:
                df = pd.DataFrame({
                    "datetime": pd.to_datetime(dates),
                    "tave": tmean
                })
                df["month_day"] = df["datetime"].dt.strftime("%m-%d")
                all_years_data.append(df[["month_day", "tave"]])

        if all_years_data:
            df_avg = pd.concat(all_years_data).groupby("month_day", as_index=False).mean()
            df_avg["sort_key"] = pd.to_datetime("2000-" + df_avg["month_day"])
            df_avg = df_avg.sort_values("sort_key").reset_index(drop=True)
            start_idx = df_avg[df_avg["month_day"] == "04-01"].index[0]
            df_avg = pd.concat([df_avg.iloc[start_idx:], df_avg.iloc[:start_idx]]).drop(columns=["sort_key"]).reset_index(drop=True)
            df_avg.rename(columns={"tave":"tave_avg"}, inplace=True)
            
            df_avg["tave_avg"] = df_avg["tave_avg"].round(2)
        else:
            df_avg = pd.DataFrame(columns=["month_day", "tave_avg"])

        # --- ② 対象期間の実測値取得 ---
        start_this = ct1_start
        end_this = max(ct1_end, forecast_end)
        
        df_this = pd.DataFrame({"date": pd.date_range(start=start_this, end=end_this).date})
        
        fetch_start = start_this
        fetch_end = min(end_this, yesterday)
        
        df_api = pd.DataFrame(columns=["date", "tave_real", "tmax_real", "tmin_real", "prcp_real"])
        
        if fetch_start <= fetch_end:
            sy, ey = fetch_start.strftime("%Y-%m-%d"), fetch_end.strftime("%Y-%m-%d")
            dates, temp_this = fetch_point_series_bulk("TMP_mea", sy, ey, lat, lon)
            _, tmax_this     = fetch_point_series_bulk("TMP_max", sy, ey, lat, lon)
            _, tmin_this     = fetch_point_series_bulk("TMP_min", sy, ey, lat, lon)
            _, prcp_this     = fetch_point_series_bulk("APCPRA",  sy, ey, lat, lon)
            
            if dates:
                df_api = pd.DataFrame({
                    "date"      : dates,
                    "tave_real" : temp_this, 
                    "tmax_real" : tmax_this, 
                    "tmin_real" : tmin_this, 
                    "prcp_real" : prcp_this
                })    

        df_this = df_this.merge(df_api, on="date", how="left")

        def assign_tag(d):
            if d <= yesterday: return "past"
            elif d <= forecast_end: return "forecast"
            else: return "normal"
                
        if not df_this.empty:
            df_this["tag"] = df_this["date"].map(assign_tag)
            df_this["month_day"] = pd.to_datetime(df_this["date"]).dt.strftime("%m-%d")
            df_this = df_this.merge(df_avg, on="month_day", how="left")
            
            df_this["tave_this"] = df_this["tave_real"].fillna(df_this["tave_avg"])
            df_this["tave_this"] = df_this["tave_this"].fillna(10.0).round(1) 
            
            df_this["prcp_this"] = df_this["prcp_real"].fillna(0.0).round(1)
            
            df_this["tmax_this"] = df_this["tmax_real"].round(1)
            df_this["tmin_this"] = df_this["tmin_real"].round(1)

            df_this.drop(columns=["month_day", "tave_avg", "tave_real", "tmax_real", "tmin_real", "prcp_real"], inplace=True)
            df_forecast = df_this.loc[df_this["tag"] == "forecast"].copy().reset_index(drop=True)
            df_forecast["date"] = df_forecast["date"].map(lambda d: d.isoformat())
        else:
            df_forecast = pd.DataFrame()

        # --- ③ 積算処理 (B6～B7の期間のみ) ---
        if not df_this.empty:
            mask_ct1 = (df_this["date"] >= ct1_start) & (df_this["date"] <= ct1_end)
            df_ct1 = df_this.loc[mask_ct1].copy().reset_index(drop=True)
        else:
            df_ct1 = pd.DataFrame()

        if df_ct1.empty:
            closest_dict, corrected_dict, hist_dict = {}, {}, {}
        else:
            df_ct1["daily_ct"] = (df_ct1["tave_this"] - threshold).clip(lower=0).round(1)          
            df_ct1["cum_ct"] = df_ct1["daily_ct"].cumsum().round(1)
            df_ct1["daily_pr"] = df_ct1["prcp_this"].round(1)
            df_ct1["cum_pr"] = df_ct1["daily_pr"].cumsum().round(1)
            
            mask_fut_rain = df_ct1["date"] > yesterday
            df_ct1.loc[mask_fut_rain, ["daily_pr", "cum_pr"]] = np.nan

            row_close = df_ct1.loc[(df_ct1["cum_ct"] - gdd1_target).abs().idxmin()]
            closest_dict = {"date": row_close["date"].isoformat(), "cum_ct": round(row_close["cum_ct"], 1)}
            
            df_ct1["corrected_cum_ct"] = df_ct1["cum_ct"] + hosei
            row_corr = df_ct1.loc[(df_ct1["corrected_cum_ct"] - gdd1_target).abs().idxmin()]
            corrected_dict = {"date": row_corr["date"].isoformat(), "cum_ct": round(row_corr["corrected_cum_ct"], 1)}

            mask_hist = df_ct1["date"] <= yesterday
            if mask_hist.any():
                row_hist = df_ct1.loc[mask_hist].iloc[-1]
                hist_dict = {"date": row_hist["date"].isoformat(), "cum_ct": round(row_hist["cum_ct"], 1), "cum_pr": round(row_hist["cum_pr"], 1)}
            else:
                hist_dict = {"date": None, "cum_ct": None, "cum_pr": None}

        def clean_json(data):
            if isinstance(data, list): return [clean_json(x) for x in data]
            if isinstance(data, dict): return {k: clean_json(v) for k, v in data.items()}
            if pd.isna(data): return "" 
            if isinstance(data, (date, datetime)): return data.isoformat()
            return data
                
        # 🚀【トークン金庫化②】最新のトークンをシステム変数から取り出してGASへ送り返す
        outgoing_token = os.environ.get("AMD_TOKEN_JSON", "")
                
        return jsonify(clean_json({
            "average": df_avg.to_dict(orient="records"), "this_year": df_this.to_dict(orient="records"),
            "ct1": df_ct1.to_dict(orient="records"), "gdd1_target": closest_dict,
            "gdd1_target_corr": corrected_dict, "ct1_until_yesterday": hist_dict, "forecast": df_forecast.to_dict(orient="records"),
            "updated_token": outgoing_token  # ★追加
        }))

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
