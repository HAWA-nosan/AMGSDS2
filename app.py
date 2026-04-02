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
        ct1_start = datetime.fromisoformat(d["ct1_start"]).date()  
        ct1_end   = datetime.fromisoformat(d["ct1_end"]).date()
        
        today = datetime.utcnow().date()
        
        # 基準年の設定（平年値と実測値の分離）
        current_year = today.year if today.month >= 4 else today.year - 1
        start_year_for_avg = current_year - 3
        target_year = ct1_start.year if ct1_start.month >= 4 else ct1_start.year - 1

        # 1. 平年値の計算
        all_years_data = []
        for year in range(start_year_for_avg, start_year_for_avg + 3):
            start = f"{year}-04-01"
            end = f"{year+1}-03-31"
            try:
                temp, tim, *_ = amd.GetMetData("TMP_mea", [start, end], [lat, lat, lon, lon])
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

        # 2. 実測値と降水量の取得（対象年度：スプレッドシート指定の年）
        start_this = f"{target_year}-04-01"
        end_this = f"{target_year + 1}-03-31"
        try:
            temp_this, tim_this, *_ = amd.GetMetData("TMP_mea", [start_this, end_this], [lat, lat, lon, lon])
            prcp_this, *_ = amd.GetMetData("APCPRA",  [start_this, end_this], [lat, lat, lon, lon])
            df_this = pd.DataFrame({
                "date"      : pd.to_datetime(tim_this).map(lambda d: d.date()),
                "tave_this" : temp_this[:, 0, 0],
                "prcp_this" : prcp_this[:, 0, 0]       
            })    
        except Exception:
            df_this = pd.DataFrame(columns=["date", "tave_this", "prcp_this", "tag"])

        yesterday = today - timedelta(days=1)
        forecast_end = today + timedelta(days=26)

        def assign_tag(d):
            if d <= yesterday: return "past"
            elif d <= forecast_end: return "forecast"
            else: return "normal"

        if not df_this.empty:
            df_this["tag"] = df_this["date"].map(assign_tag)
            
            df_this["month_day"] = df_this["date"].map(lambda d: d.strftime("%m-%d"))
            if not df_avg.empty:
                df_this = df_this.merge(df_avg[["month_day", "tave_avg"]], on="month_day", how="left")
                mask = df_this["tag"] == "normal"
                df_this.loc[mask, "tave_this"] = df_this.loc[mask, "tave_avg"]
                df_this.drop(columns=["month_day", "tave_avg"], inplace=True)
            else:
                df_this.drop(columns=["month_day"], inplace=True)

        # ★ 予報部分の修正：現実の今年度データを丸ごと取得して、9日間だけ切り出す！
        real_start = f"{current_year}-04-01"
        real_end   = f"{current_year + 1}-03-31"
        try:
            r_temp, r_tim, *_ = amd.GetMetData("TMP_mea", [real_start, real_end], [lat, lat, lon, lon])
            r_prcp, *_ = amd.GetMetData("APCPRA", [real_start, real_end], [lat, lat, lon, lon])
            df_real = pd.DataFrame({
                "date": pd.to_datetime(r_tim).map(lambda d: d.date()),
                "tave_this": r_temp[:, 0, 0],
                "prcp_this": r_prcp[:, 0, 0]
            })
            
            # 昨日から7日後までの9日間を抽出
            f_start_date = today - timedelta(days=1)
            f_end_date   = today + timedelta(days=7)
            mask_f = (df_real["date"] >= f_start_date) & (df_real["date"] <= f_end_date)
            df_forecast = df_real.loc[mask_f].reset_index(drop=True)
        except Exception:
            df_forecast = pd.DataFrame(columns=["date", "tave_this", "prcp_this"])

        # 3. 積算範囲 (CT1) の計算
        if not df_this.empty:
            mask = (df_this["date"] >= ct1_start) & (df_this["date"] <= ct1_end)
            df_ct1 = df_this.loc[mask].copy().reset_index(drop=True)
        else:
            df_ct1 = pd.DataFrame()

        if df_ct1.empty:
            closest_dict = {}
            corrected_dict = {}
            hist_dict = {}
        else:
            df_ct1["daily_ct"] = (df_ct1["tave_this"] - threshold).clip(lower=0).round(1)          
            df_ct1["cum_ct"] = df_ct1["daily_ct"].cumsum().round(1)
            df_ct1["daily_pr"] = df_ct1["prcp_this"].round(1)
            df_ct1["cum_pr"] = df_ct1["daily_pr"].cumsum().round(1)

            df_ct1["abs_diff"] = (df_ct1["cum_ct"] - gdd1_target).abs()
            row_close = df_ct1.loc[df_ct1["abs_diff"].idxmin()]
            closest_dict = {
                "date": row_close["date"].isoformat(),
                "cum_ct": round(row_close["cum_ct"], 1)
            }

            df_ct1["corrected_cum_ct"] = df_ct1["cum_ct"] + hosei
            df_ct1["abs_diff_corr"] = (df_ct1["corrected_cum_ct"] - gdd1_target).abs()
            row_corr = df_ct1.loc[df_ct1["abs_diff_corr"].idxmin()]
            corrected_dict = {
                "date": row_corr["date"].isoformat(),
                "cum_ct": round(row_corr["corrected_cum_ct"], 1)
            }

            mask_hist = df_ct1["date"] <= yesterday
            if mask_hist.any():
                row_hist = df_ct1.loc[mask_hist].iloc[-1]
                hist_dict = {
                    "cum_ct": round(row_hist["cum_ct"], 1),
                    "cum_pr": round(row_hist["cum_pr"], 1)
                }
            else:
                hist_dict = {}

        # 4. JSONデータのクリーンアップ
        def replace_nan(d):
            if isinstance(d, list): return [replace_nan(x) for x in d]
            if isinstance(d, dict): return {k: replace_nan(v) for k, v in d.items()}
            if isinstance(d, float) and math.isnan(d): return None
            return d

        if not df_this.empty: df_this["date"] = df_this["date"].map(lambda d: d.isoformat())
        if not df_forecast.empty: df_forecast["date"] = df_forecast["date"].map(lambda d: d.isoformat())
        if not df_ct1.empty: df_ct1["date"] = df_ct1["date"].map(lambda d: d.isoformat())

        return jsonify({
            "average": replace_nan(df_avg.to_dict(orient="records")),
            "this_year": replace_nan(df_this.to_dict(orient="records")),
            "forecast": replace_nan(df_forecast.to_dict(orient="records")),
            "ct1": replace_nan(df_ct1.to_dict(orient="records")),
            "gdd1_target": closest_dict,
            "gdd1_target_corr": corrected_dict,
            "ct1_until_yesterday": hist_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
