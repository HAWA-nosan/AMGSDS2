# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import traceback
import AMD_Tools4 as amd

app = Flask(__name__)

# エラー時にJSONを返す安全装置
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"status": "error", "message": "Server Crash", "trace": traceback.format_exc()}), 500

@app.route("/get_temp", methods=["POST"])
def get_climate_data():
    try:
        d = request.get_json()
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
            try:
                temp, tim, *_ = amd.GetMetData("TMP_mea", [start, end], [lat, lat, lon, lon])
                df = pd.DataFrame({
                    "datetime": pd.to_datetime(list(tim)),
                    "tave": temp[:, 0, 0]
                })
                df["month_day"] = df["datetime"].dt.strftime("%m-%d")
                all_years_data.append(df[["month_day", "tave"]])
            except Exception: pass

        if all_years_data:
            df_avg = pd.concat(all_years_data).groupby("month_day", as_index=False).mean()
            df_avg["sort_key"] = pd.to_datetime("2000-" + df_avg["month_day"])
            df_avg = df_avg.sort_values("sort_key").reset_index(drop=True)
            start_idx = df_avg[df_avg["month_day"] == "04-01"].index[0]
            df_avg = pd.concat([df_avg.iloc[start_idx:], df_avg.iloc[:start_idx]]).drop(columns=["sort_key"]).reset_index(drop=True)
            df_avg.rename(columns={"tave":"tave_avg"}, inplace=True)
            
            # ★ ここを追加！過去3年平均を「小数点2桁」で丸める
            df_avg["tave_avg"] = df_avg["tave_avg"].round(2)
        else:
            df_avg = pd.DataFrame(columns=["month_day", "tave_avg"])

        # --- ② 対象期間の実測値取得 ---
        start_this = ct1_start
        end_this = max(ct1_end, forecast_end)
        
        df_this = pd.DataFrame({"date": pd.date_range(start=start_this, end=end_this).date})
        try:
            sy, ey = start_this.strftime("%Y-%m-%d"), end_this.strftime("%Y-%m-%d")
            temp_this, tim_this, *_ = amd.GetMetData("TMP_mea", [sy, ey], [lat, lat, lon, lon])
            tmax_this, _, *_        = amd.GetMetData("TMP_max", [sy, ey], [lat, lat, lon, lon])
            tmin_this, _, *_        = amd.GetMetData("TMP_min", [sy, ey], [lat, lat, lon, lon])
            prcp_this, _, *_        = amd.GetMetData("APCPRA",  [sy, ey], [lat, lat, lon, lon])
            df_api = pd.DataFrame({
                "date"      : pd.to_datetime(list(tim_this)).date,
                "tave_real" : temp_this[:, 0, 0], 
                "tmax_real" : tmax_this[:, 0, 0], 
                "tmin_real" : tmin_this[:, 0, 0], 
                "prcp_real" : prcp_this[:, 0, 0]
            })    
        except Exception:
            df_api = pd.DataFrame(columns=["date", "tave_real", "tmax_real", "tmin_real", "prcp_real"])

        df_this = df_this.merge(df_api, on="date", how="left")

        def assign_tag(d):
            if d <= yesterday: return "past"
            elif d <= forecast_end: return "forecast"
            else: return "normal"
                
        if not df_this.empty:
            df_this["tag"] = df_this["date"].map(assign_tag)
            df_this["month_day"] = pd.to_datetime(df_this["date"]).dt.strftime("%m-%d")
            df_this = df_this.merge(df_avg, on="month_day", how="left")
            
            mask = df_this["tag"] == "normal"
            
            df_this["tave_this"] = np.where(mask, df_this["tave_avg"], df_this["tave_real"])
            df_this["prcp_this"] = np.where(mask, 0.0, df_this["prcp_real"])
            
            # P列の平均気温・降水量は1桁で統一
            df_this["tave_this"] = df_this["tave_this"].fillna(10.0).round(1)
            df_this["prcp_this"] = df_this["prcp_this"].fillna(0.0).round(1)
            
            # 最高・最低気温は実測値(real)をそのまま使う
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
            
            # 将来の雨量は積算しない
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

        # JSONの空欄（NaN）を空白文字に変換する処理
        def clean_json(data):
            if isinstance(data, list): return [clean_json(x) for x in data]
            if isinstance(data, dict): return {k: clean_json(v) for k, v in data.items()}
            if pd.isna(data): return "" 
            if isinstance(data, (date, datetime)): return data.isoformat()
            return data
                
        return jsonify(clean_json({
            "average": df_avg.to_dict(orient="records"), "this_year": df_this.to_dict(orient="records"),
            "ct1": df_ct1.to_dict(orient="records"), "gdd1_target": closest_dict,
            "gdd1_target_corr": corrected_dict, "ct1_until_yesterday": hist_dict, "forecast": df_forecast.to_dict(orient="records")
        }))

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
