# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import traceback

try:
    import AMD_Tools4 as amd
except ImportError:
    amd = None

app = Flask(__name__)

# 万が一のエラー時に500エラー画面ではなくJSONを返す安全装置
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"status": "error", "message": "Server Crash", "trace": traceback.format_exc()}), 500

@app.route("/get_temp", methods=["POST"])
def get_climate_data():
    try:
        d = request.get_json()
        lat, lon = map(float, (d["lat"], d["lon"]))
        threshold = float(d["threshold"])
        gdd1_target = float(d["gdd1"])
        hosei = float(d.get("hosei", 0.0))
        
        ct1_start = datetime.fromisoformat(d["ct1_start"]).date()  
        ct1_end   = datetime.fromisoformat(d["ct1_end"]).date()
        
        today = datetime.utcnow().date()
        
        real_this_year = today.year if today.month >= 4 else today.year - 1
        target_year = ct1_start.year if ct1_start.month >= 4 else ct1_start.year - 1

        all_years_data = []
        for year in range(real_this_year - 3, real_this_year):
            start = f"{year}-04-01"
            end = f"{year+1}-03-31"
            try:
                temp, tim, *_ = amd.GetMetData("TMP_mea", [start, end], [lat, lat, lon, lon])
                flat_temp = temp[:, 0, 0]

                s_dates = pd.to_datetime(pd.Series(list(tim)))
                df = pd.DataFrame({
                    "datetime": s_dates,
                    "tave": flat_temp
                })
                df["month_day"] = df["datetime"].dt.strftime("%m-%d")
                all_years_data.append(df[["month_day", "tave"]])
            except Exception:
                pass

        if all_years_data:
            df_concat = pd.concat(all_years_data)
            df_avg = df_concat.groupby("month_day", as_index=False)["tave"].mean()
            df_avg.rename(columns={"tave": "tave_avg"}, inplace=True)

            def reorder_from_april(df):
                df = df.copy()
                df["sort_key"] = pd.to_datetime("2000-" + df["month_day"])
                df = df.sort_values("sort_key").reset_index(drop=True)
                start_idx = df[df["month_day"] == "04-01"].index[0]
                return pd.concat([df.iloc[start_idx:], df.iloc[:start_idx]]).drop(columns=["sort_key"]).reset_index(drop=True)

            df_avg = reorder_from_april(df_avg)
            df_avg["tave_avg"] = df_avg["tave_avg"].round(1)
        else:
            df_avg = pd.DataFrame(columns=["month_day", "tave_avg"])

        start_this = f"{target_year}-04-01"
        end_this = f"{target_year + 1}-03-31"
        
        df_this = pd.DataFrame({"date": pd.date_range(start=start_this, end=end_this).date})
        
        try:
            temp_this, tim_this, *_ = amd.GetMetData("TMP_mea", [start_this, end_this], [lat, lat, lon, lon])
            prcp_this, *_ = amd.GetMetData("APCPRA",  [start_this, end_this], [lat, lat, lon, lon])
            
            s_this_dates = pd.to_datetime(pd.Series(list(tim_this)))
            df_api = pd.DataFrame({
                "date"      : s_this_dates.dt.date,
                "tave_real" : temp_this[:, 0, 0],
                "prcp_real" : prcp_this[:, 0, 0]       
            })    
        except Exception:
            df_api = pd.DataFrame(columns=["date", "tave_real", "prcp_real"])

        df_this = df_this.merge(df_api, on="date", how="left")

        yesterday = today - timedelta(days=1)
        forecast_end = today + timedelta(days=26)

        def assign_tag(d):
            if d <= yesterday:
                return "past"
            elif d <= forecast_end:
                return "forecast"
            else:
                return "normal"
                
        if not df_this.empty:
            df_this["tag"] = df_this["date"].map(assign_tag)
            
            df_this["month_day"] = pd.to_datetime(df_this["date"]).dt.strftime("%m-%d")
            df_this = df_this.merge(df_avg[["month_day", "tave_avg"]], on="month_day", how="left")
            
            mask_normal = df_this["tag"] == "normal"
            df_this["tave_this"] = df_this["tave_real"]
            df_this.loc[mask_normal, "tave_this"] = df_this.loc[mask_normal, "tave_avg"]
            
            df_this["prcp_this"] = df_this["prcp_real"]
            df_this.loc[mask_normal, "prcp_this"] = 0.0
            
            df_this["tave_this"] = df_this["tave_this"].fillna(10.0).round(1)
            df_this["prcp_this"] = df_this["prcp_this"].fillna(0.0).round(1)

            df_this.drop(columns=["month_day", "tave_avg", "tave_real", "prcp_real"], inplace=True)
            
            df_forecast = df_this.loc[df_this["tag"] == "forecast"].reset_index(drop=True)
            df_forecast["date"] = df_forecast["date"].map(lambda d: d.isoformat() if isinstance(d, date) else d)
        else:
            df_forecast = pd.DataFrame()

        if not df_this.empty:
            mask = (df_this["date"] >= ct1_start) & (df_this["date"] <= ct1_end)
            df_ct1_period = df_this.loc[mask].reset_index(drop=True)
        else:
            df_ct1_period = pd.DataFrame()

        df_ct1 = df_ct1_period.copy()
        
        if df_ct1.empty:
            closest_dict = {}
            corrected_dict = {}
            hist_dict = {}
        else:
            df_ct1["daily_ct"] = (df_ct1["tave_this"] - threshold).clip(lower=0).round(1)          
            df_ct1["cum_ct"] = df_ct1["daily_ct"].cumsum().round(1)
            df_ct1["daily_pr"] = df_ct1["prcp_this"].round(1)
            df_ct1["cum_pr"] = df_ct1["daily_pr"].cumsum().round(1)

            mask_fut = df_ct1["date"] > forecast_end
            df_ct1.loc[mask_fut, "daily_pr"] = np.nan
            df_ct1.loc[mask_fut, "cum_pr"] = np.nan

            df_ct1["abs_diff"] = (df_ct1["cum_ct"] - gdd1_target).abs()
            idx_closest = df_ct1["abs_diff"].idxmin()
            row_close   = df_ct1.loc[idx_closest]

            closest_dict = {
                "date"     : row_close["date"].isoformat() if isinstance(row_close["date"], date) else row_close["date"],
                "cum_ct"   : round(row_close["cum_ct"], 1),
                "abs_diff" : round(row_close["abs_diff"], 1)
            }

            df_ct1["corrected_cum_ct"] = df_ct1["cum_ct"] + hosei
            df_ct1["abs_diff_corr"] = (df_ct1["corrected_cum_ct"] - gdd1_target).abs()
            idx_corr  = df_ct1["abs_diff_corr"].idxmin()
            row_corr  = df_ct1.loc[idx_corr]

            corrected_dict = {
                "date"        : row_corr["date"].isoformat() if isinstance(row_corr["date"], date) else row_corr["date"],
                "cum_ct"      : round(row_corr["corrected_cum_ct"], 1),
                "abs_diff"    : round(row_corr["abs_diff_corr"], 1),
            }

            mask_hist = df_ct1["date"] <= yesterday
            if mask_hist.any():
                row_hist = df_ct1.loc[mask_hist].iloc[-1]
                hist_dict = {
                    "date"   : row_hist["date"].isoformat() if isinstance(row_hist["date"], date) else row_hist["date"],
                    "cum_ct" : round(row_hist["cum_ct"], 1),
                    "cum_pr" : round(row_hist["cum_pr"], 1)
                }
            else:
                hist_dict = {"date": None, "cum_ct": None, "cum_pr": None}

        def replace_nan_with_none(data):
            if isinstance(data, list):
                return [replace_nan_with_none(x) for x in data]
            elif isinstance(data, dict):
                return {k: replace_nan_with_none(v) for k, v in data.items()}
            elif pd.isna(data):
                return ""
            else:
                return data
                
        if not df_this.empty: df_this["date"] = df_this["date"].map(lambda d: d.isoformat() if isinstance(d, date) else d)   
        if not df_ct1_period.empty: df_ct1_period["date"] = df_ct1_period["date"].map(lambda d: d.isoformat() if isinstance(d, date) else d) 
        if not df_ct1.empty: df_ct1["date"] = df_ct1["date"].map(lambda d: d.isoformat() if isinstance(d, date) else d) 
        
        df_avg_clean = replace_nan_with_none(df_avg.to_dict(orient="records"))
        df_this_clean = replace_nan_with_none(df_this.to_dict(orient="records"))
        df_ct1_period_clean = replace_nan_with_none(df_ct1_period.to_dict(orient="records"))
        df_ct1_clean = replace_nan_with_none(df_ct1.to_dict(orient="records"))
        df_forecast_clean = replace_nan_with_none(df_forecast.to_dict(orient="records")) if not df_forecast.empty else []

        return jsonify({
            "average": df_avg_clean,
            "this_year": df_this_clean,
            "ct1_period": df_ct1_period_clean,
            "ct1"       : df_ct1_clean,
            "gdd1_target": closest_dict,
            "gdd1_target_corr"   : corrected_dict,      
            "ct1_until_yesterday": hist_dict,            
            "forecast": df_forecast_clean
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
