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
# ★ 爆速化の要（キャッシュシステム）
# 過去のデータは変化しないため、一度取得したものはメモリに暗記させます。
# これにより、複数シートを一括更新する際の通信時間を極限まで削ります！
# ====================================================================
@lru_cache(maxsize=128)
def get_cached_avg_data(year, lat, lon):
    start_str = f"{year}-04-01"
    end_str = f"{year+1}-03-31"
    try:
        temp, tim, *_ = amd.GetMetData("TMP_mea", [start_str, end_str], [lat, lat, lon, lon])
        df = pd.DataFrame({
            "datetime": pd.to_datetime(tim),
            "tave": temp[:, 0, 0]
        })
        df["month_day"] = df["datetime"].dt.strftime("%m-%d")
        return df[["month_day", "tave"]]
    except Exception:
        return None

@lru_cache(maxsize=64)
def get_cached_past_year(py_start, py_end, lat, lon):
    try:
        t_py, tim_py, *_ = amd.GetMetData("TMP_mea", [py_start, py_end], [lat, lat, lon, lon])
        p_py, *_       = amd.GetMetData("APCPRA", [py_start, py_end], [lat, lat, lon, lon])
        return pd.DataFrame({
            "date": pd.to_datetime(tim_py).dt.normalize(),
            "tave_real": t_py[:, 0, 0],
            "prcp_real": p_py[:, 0, 0]
        })
    except Exception:
        return None
# ====================================================================

@app.route("/get_temp", methods=["POST"])
def get_climate_data():
    try:
        d = request.get_json()
        lat, lon = map(float, (d["lat"], d["lon"]))
        threshold = float(d["threshold"])
        gdd1_target = float(d["gdd1"])
        hosei = float(d.get("hosei", 0.0))
        
        ct1_start = pd.to_datetime(d["ct1_start"])
        ct1_end   = pd.to_datetime(d["ct1_end"])
        
        today = pd.Timestamp(datetime.utcnow().date())
        yesterday = today - pd.Timedelta(days=1)
        forecast_end = today + pd.Timedelta(days=26)
        
        # 基準年の設定
        current_year = today.year if today.month >= 4 else today.year - 1
        start_year_for_avg = current_year - 3
        target_year = ct1_start.year if ct1_start.month >= 4 else ct1_start.year - 1

        # 1. 平年値の計算（キャッシュを利用して爆速化）
        all_years_data = []
        for year in range(start_year_for_avg, start_year_for_avg + 3):
            res = get_cached_avg_data(year, lat, lon)
            if res is not None:
                all_years_data.append(res.copy()) # 安全のためコピー

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
        
        # 昨年度データ（キャッシュを利用して爆速化）
        py_start = f"{current_year-1}-04-01"
        py_end   = f"{current_year}-03-31"
        df_py = get_cached_past_year(py_start, py_end, lat, lon)
        if df_py is not None:
            df_available_list.append(df_py.copy())

        # 今年度データ（最新の予報が含まれるため、これだけは毎回取得する）
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

        # 3. 指定された年度の「完璧に織り交ぜられた」365日予測タイムラインを作成
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

        # 5. 積算範囲 (CT1) の計算
        mask_ct = (df_this["date"] >= ct1_start) & (df_this["date"] <= ct1_end)
        df_ct1 = df_this.loc[mask_ct].copy().reset_index(drop=True)

        if df_ct1.empty:
            closest_dict, corrected_dict, hist_dict = {}, {}, {}
        else:
            df_ct1["daily_ct"] = (df_ct1["tave_this"] - threshold).clip(lower=0).round(1)          
            df_ct1["cum_ct"] = df_ct1["daily_ct"].cumsum().round(1)
            df_ct1["daily_pr"] = df_ct1["prcp_this"].round(1)
            df_ct1["cum_pr"] = df_ct1["daily_pr"].cumsum().round(1)

            df_ct1["abs_diff"] = (df_ct1["cum_ct"] - gdd1_target).abs()
            row_close = df_ct1.loc[df_ct1["abs_diff"].idxmin()]
            closest_dict = {
                "date": row_close["date"].strftime("%Y-%m-%d"),
                "cum_ct": round(row_close["cum_ct"], 1)
            }

            df_ct1["corrected_cum_ct"] = df_ct1["cum_ct"] + hosei
            df_ct1["abs_diff_corr"] = (df_ct1["corrected_cum_ct"] - gdd1_target).abs()
            row_corr = df_ct1.loc[df_ct1["abs_diff_corr"].idxmin()]
            corrected_dict = {
                "date": row_corr["date"].strftime("%Y-%m-%d"),
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

        # 6. JSONデータのクリーンアップ
        def replace_nan(d):
            if isinstance(d, list): return [replace_nan(x) for x in d]
            if isinstance(d, dict): return {k: replace_nan(v) for k, v in d.items()}
            if isinstance(d, float) and math.isnan(d): return None
            return d

        df_this["date"] = df_this["date"].dt.strftime("%Y-%m-%d")
        df_forecast["date"] = df_forecast["date"].dt.strftime("%Y-%m-%d")
        if not df_ct1.empty: df_ct1["date"] = df_ct1["date"].dt.strftime("%Y-%m-%d")

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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
