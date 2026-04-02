# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta, date
from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np
import math
import traceback

try:
    import AMD_Tools4 as amd
except ImportError:
    amd = None

app = Flask(__name__)

DATE_FMT = "%Y-%m-%d"
DL_CSV_PATH = Path(__file__).resolve().parent / "sweetcorn_data-DL.csv"

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"status": "error", "message": "Server Crash", "trace": traceback.format_exc()}), 500

def to_date(s: str) -> date:
    if not s: return None
    return datetime.fromisoformat(s).date()

def parse_float(value, allow_none=False):
    if value is None or value == "": return None if allow_none else 0.0
    return float(value)

def parse_int(value):
    return 0 if value is None or value == "" else int(float(value))

def round_or_none(x, ndigits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
    return round(float(x), ndigits)

def replace_nan_with_none(data):
    if isinstance(data, list): return [replace_nan_with_none(x) for x in data]
    if isinstance(data, dict): return {k: replace_nan_with_none(v) for k, v in data.items()}
    if pd.isna(data): return ""
    if isinstance(data, (np.integer, np.floating)): return float(data) if not np.isnan(data) else ""
    return data

def to_iso_or_none(d):
    if pd.isna(d): return ""
    if isinstance(d, (datetime, date, pd.Timestamp)): return d.strftime("%Y-%m-%d")
    return d

def parse_request_payload(d: dict) -> dict:
    return {
        "lat": float(d["lat"]), "lon": float(d["lon"]),
        "ct1_start": to_date(d["ct1_start"]), "ct1_end": to_date(d.get("ct1_end", "")),
        "method1": parse_int(d.get("method1", 1)), "base_threshold1": parse_float(d.get("base_threshold1", 0)),
        "ceiling_threshold1": parse_float(d.get("ceiling_threshold1"), allow_none=True), "gdd1_target": parse_float(d.get("gdd1_target", 0)),
        "ct2_start": to_date(d.get("ct2_start", "")), "ct2_end": to_date(d.get("ct2_end", "")),
        "method2": parse_int(d.get("method2", 1)), "base_threshold2": parse_float(d.get("base_threshold2", 0)),
        "ceiling_threshold2": parse_float(d.get("ceiling_threshold2"), allow_none=True), "gdd2_target": parse_float(d.get("gdd2_target", 0)),
    }

@lru_cache(maxsize=256)
def fetch_point_series(var_name: str, start_date: str, end_date: str, lat: float, lon: float):
    try:
        arr, tim, *_ = amd.GetMetData(var_name, [start_date, end_date], [lat, lat, lon, lon])
        values = arr[:, 0, 0]
        s_dates = pd.to_datetime(pd.Series(list(tim)))
        dates = s_dates.dt.normalize().dt.date.tolist()
        return dates, list(values)
    except Exception:
        return [], []

def build_average_temperature(lat: float, lon: float, fiscal_year: int, n_years: int = 3) -> pd.DataFrame:
    all_years = []
    start_year = fiscal_year - n_years
    for year in range(start_year, fiscal_year):
        start, end = f"{year}-04-01", f"{year + 1}-03-31"
        dates, tmean = fetch_point_series("TMP_mea", start, end, lat, lon)
        _, tmax = fetch_point_series("TMP_max", start, end, lat, lon)
        _, tmin = fetch_point_series("TMP_min", start, end, lat, lon)
        if dates:
            df = pd.DataFrame({"datetime": pd.to_datetime(dates), "tave": tmean, "tmax": tmax, "tmin": tmin})
            df["month_day"] = df["datetime"].dt.strftime("%m-%d")
            all_years.append(df[["month_day", "tave", "tmax", "tmin"]])

    if not all_years: return pd.DataFrame(columns=["month_day", "tave_avg", "tmax_avg", "tmin_avg"])
    df_concat = pd.concat(all_years, ignore_index=True)
    df_avg = df_concat.groupby("month_day", as_index=False).mean()
    df_avg.rename(columns={"tave": "tave_avg", "tmax": "tmax_avg", "tmin": "tmin_avg"}, inplace=True)
    df_avg["tave_avg"] = df_avg["tave_avg"].round(1)
    df_avg["tmax_avg"] = df_avg["tmax_avg"].round(1)
    df_avg["tmin_avg"] = df_avg["tmin_avg"].round(1)

    df_avg["sort_key"] = pd.to_datetime("2000-" + df_avg["month_day"])
    df_avg = df_avg.sort_values("sort_key").reset_index(drop=True)
    idx_m = df_avg.index[df_avg["month_day"] == "04-01"]
    start_idx = idx_m[0] if len(idx_m) > 0 else 0
    df_avg = pd.concat([df_avg.iloc[start_idx:], df_avg.iloc[:start_idx]], ignore_index=True).drop(columns="sort_key")
    return df_avg

def build_this_year_dataframe(lat: float, lon: float, fiscal_year: int, today: date, df_avg: pd.DataFrame) -> pd.DataFrame:
    start_this = pd.to_datetime(f"{fiscal_year}-04-01").date()
    end_this = pd.to_datetime(f"{fiscal_year + 1}-03-31").date()
    
    # 欠損日を防ぐため、365日分のカレンダー枠を完全に作る
    df = pd.DataFrame({"date": pd.date_range(start=start_this, end=end_this).date})

    yesterday, forecast_end = today - timedelta(days=1), today + timedelta(days=26)
    df["tag"] = df["date"].apply(lambda d: "past" if d <= yesterday else ("forecast" if d <= forecast_end else "normal"))

    df_avail_list = []
    for fy in [fiscal_year - 1, fiscal_year]:
        s, e = f"{fy}-04-01", f"{fy+1}-03-31"
        dates, tmean = fetch_point_series("TMP_mea", s, e, lat, lon)
        _, tmax = fetch_point_series("TMP_max", s, e, lat, lon)
        _, tmin = fetch_point_series("TMP_min", s, e, lat, lon)
        _, prcp = fetch_point_series("APCPRA", s, e, lat, lon)
        if dates:
            df_avail_list.append(pd.DataFrame({"date": dates, "tave_real": tmean, "tmax_real": tmax, "tmin_real": tmin, "prcp_real": prcp}))
    
    if df_avail_list:
        df_avail = pd.concat(df_avail_list).drop_duplicates(subset=["date"], keep="last")
    else:
        df_avail = pd.DataFrame(columns=["date", "tave_real", "tmax_real", "tmin_real", "prcp_real"])

    df = df.merge(df_avail, on="date", how="left")
    df["month_day"] = pd.to_datetime(df["date"]).dt.strftime("%m-%d")
    
    if not df_avg.empty:
        df = df.merge(df_avg[["month_day", "tave_avg", "tmax_avg", "tmin_avg"]], on="month_day", how="left")
    else:
        df["tave_avg"], df["tmax_avg"], df["tmin_avg"] = 10.0, 14.0, 6.0

    mask_normal = df["tag"] == "normal"
    for col, avg_col in [("tave", "tave_avg"), ("tmax", "tmax_avg"), ("tmin", "tmin_avg")]:
        df[f"{col}_this"] = df[f"{col}_real"]
        df.loc[mask_normal, f"{col}_this"] = np.nan
        df[f"{col}_this"] = df[f"{col}_this"].astype(float).fillna(df[avg_col]).round(1)

    df["prcp_this"] = df["prcp_real"]
    df.loc[mask_normal, "prcp_this"] = 0.0
    df["prcp_this"] = df["prcp_this"].astype(float).fillna(0.0).round(1)

    df.drop(columns=["month_day", "tave_avg", "tmax_avg", "tmin_avg", "tave_real", "tmax_real", "tmin_real", "prcp_real"], inplace=True)
    return df

def load_daylength_table(csv_path: Path = DL_CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists(): raise FileNotFoundError(f"Daylength CSV not found: {csv_path}")
    df_dl = pd.read_csv(csv_path)
    df_dl["month_day"] = df_dl["date"].astype(str).str.strip()
    df_dl["DL"] = pd.to_numeric(df_dl["DL"], errors="coerce")
    return df_dl.drop_duplicates(subset=["month_day"]).reset_index(drop=True)[["month_day", "DL"]]

def add_daylength_from_csv(df: pd.DataFrame, df_dl_master: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["DL_hours"] = pd.Series(dtype=float)
        return out
    out = df.copy()
    out["month_day"] = pd.to_datetime(out["date"]).dt.strftime("%-m/%-d")
    out = out.merge(df_dl_master, on="month_day", how="left")
    out["DL_hours"] = out["DL"].astype(float)
    return out.drop(columns=["month_day", "DL"])

def validate_method_and_thresholds(method: int, t_base: float, t_ceiling):
    if method not in range(1, 9): raise ValueError(f"Unsupported method: {method}")
    if method in [3, 4, 7, 8] and t_ceiling is None: raise ValueError(f"Method {method} requires ceiling_threshold.")
    if t_ceiling is not None and t_ceiling < t_base: raise ValueError("ceiling_threshold must be >= base_threshold.")

def calc_daily_gdd_core(tmean, tmax, method, t_base, t_ceiling=None):
    if method == 1: return max(0.0, tmean - t_base)
    if method == 2: return max(0.0, tmax - t_base)
    if method == 3:
        if tmax <= t_base: return 0.0
        val = tmax - t_base if tmax <= t_ceiling else (2 * t_ceiling - tmax) - t_base
        return max(0.0, val)
    if method == 4:
        if tmean <= t_base: return 0.0
        val = tmean - t_base if tmax <= t_ceiling else (tmean - (tmax - t_ceiling)) - t_base
        return max(0.0, val)
    return 0.0

def calc_daily_gdd(row, method, t_base, t_ceiling=None):
    if method in [1, 2, 3, 4]: return calc_daily_gdd_core(row["tave_this"], row["tmax_this"], method, t_base, t_ceiling)
    core = calc_daily_gdd_core(row["tave_this"], row["tmax_this"], method - 4, t_base, t_ceiling)
    return core * row["DL_hours"]

def build_accumulation_dataframe(df_src: pd.DataFrame, start_date: date, end_date: date, method: int, t_base: float, t_ceiling, target_gdd: float, df_dl_master: pd.DataFrame):
    if not start_date or not end_date: return pd.DataFrame(), {"date": None, "cum_ct": None, "daily_ct": None, "abs_diff": None}
    validate_method_and_thresholds(method, t_base, t_ceiling)
    if df_src.empty: return df_src.copy(), {"date": None, "cum_ct": None, "daily_ct": None, "abs_diff": None}

    mask = (df_src["date"] >= start_date) & (df_src["date"] <= end_date)
    df = df_src.loc[mask].copy().reset_index(drop=True)
    if df.empty: return df, {"date": None, "cum_ct": None, "daily_ct": None, "abs_diff": None}

    df = add_daylength_from_csv(df, df_dl_master) if method in [5, 6, 7, 8] else df.assign(DL_hours=np.nan)
    df["daily_ct"] = df.apply(lambda row: calc_daily_gdd(row, method, t_base, t_ceiling), axis=1).round(1)
    df["cum_ct"] = df["daily_ct"].cumsum().round(1)

    df["daily_pr"] = df["prcp_this"].round(1)
    df["cum_pr"] = df["daily_pr"].cumsum().round(1)
    
    # ★ 未来の雨量は完全に消す
    forecast_end_date = datetime.utcnow().date() + timedelta(days=26)
    mask_future = df["date"] > forecast_end_date
    df.loc[mask_future, ["daily_pr", "cum_pr"]] = np.nan

    try:
        df["abs_diff"] = (df["cum_ct"] - target_gdd).abs().round(1)
        row_close = df.loc[df["abs_diff"].idxmin()]
        closest = {
            "date": row_close["date"].isoformat(), "cum_ct": round_or_none(row_close["cum_ct"], 1),
            "daily_ct": round_or_none(row_close["daily_ct"], 1), "abs_diff": round_or_none(row_close["abs_diff"], 1)
        }
    except Exception:
        closest = {"date": None, "cum_ct": None, "daily_ct": None, "abs_diff": None}
        
    return df, closest

def make_hist_dict_simple_ct(date_start: date, date_end: date, df_src: pd.DataFrame):
    if df_src.empty or not date_start or date_end < date_start: return {"date": None, "cum_ct": None, "cum_pr": None}
    mask = (df_src["date"] >= date_start) & (df_src["date"] <= date_end)
    if not mask.any(): return {"date": None, "cum_ct": None, "cum_pr": None}
    tmp = df_src.loc[mask].copy()
    tmp["daily_ct"] = tmp["tave_this"].clip(lower=0)
    return {"date": date_end.isoformat(), "cum_ct": round_or_none(tmp["daily_ct"].sum(), 1), "cum_pr": round_or_none(tmp["prcp_this"].sum(), 1)}

def dataframe_to_records_with_iso_date(df: pd.DataFrame):
    out = df.copy()
    if "date" in out.columns: out["date"] = out["date"].map(to_iso_or_none)
    return replace_nan_with_none(out.to_dict(orient="records"))

@app.route("/get_temp", methods=["POST"])
def get_climate_data():
    try:
        d = request.get_json()
        params = parse_request_payload(d)
        today = datetime.utcnow().date()
        fiscal_year_for_avg = today.year if today.month >= 4 else today.year - 1
        target_fiscal_year = params["ct1_start"].year if params["ct1_start"].month >= 4 else params["ct1_start"].year - 1
        yesterday, forecast_end = today - timedelta(days=1), today + timedelta(days=26)

        try: df_dl_master = load_daylength_table()
        except Exception: df_dl_master = pd.DataFrame(columns=["month_day", "DL"])

        df_avg = build_average_temperature(params["lat"], params["lon"], fiscal_year_for_avg, n_years=3)
        df_this = build_this_year_dataframe(params["lat"], params["lon"], target_fiscal_year, today, df_avg)
        
        mask_f = (df_this["date"] >= (today - timedelta(days=1))) & (df_this["date"] <= (today + timedelta(days=7)))
        df_forecast = df_this.loc[mask_f].copy().reset_index(drop=True) if mask_f.any() else pd.DataFrame()

        df_ct1, closest1 = build_accumulation_dataframe(df_this, params["ct1_start"], params["ct1_end"], params["method1"], params["base_threshold1"], params["ceiling_threshold1"], params["gdd1_target"], df_dl_master)
        
        # B13が空欄だった場合はCT1の達成日を引き継ぐ
        ct2_start_calc = params["ct2_start"] if params["ct2_start"] else (to_date(closest1["date"]) if closest1.get("date") else today)
        ct2_end_calc = params["ct2_end"] if params["ct2_end"] else pd.to_datetime(f"{target_fiscal_year+1}-03-31").date()
        
        df_ct2, closest2 = build_accumulation_dataframe(df_this, ct2_start_calc, ct2_end_calc, params["method2"], params["base_threshold2"], params["ceiling_threshold2"], params["gdd2_target"], df_dl_master)

        hist_dict1 = make_hist_dict_simple_ct(params["ct1_start"], yesterday, df_this)
        hist_dict2 = make_hist_dict_simple_ct(ct2_start_calc, yesterday, df_this)

        df_this_json, df_forecast_json = df_this.copy(), df_forecast.copy()
        if not df_this_json.empty: df_this_json["date"] = df_this_json["date"].map(to_iso_or_none)
        if not df_forecast_json.empty: df_forecast_json["date"] = df_forecast_json["date"].map(to_iso_or_none)

        ct1_period_df = df_this.loc[(df_this["date"] >= params["ct1_start"]) & (df_this["date"] <= params["ct1_end"])].reset_index(drop=True) if not df_this.empty else pd.DataFrame()
        ct2_period_df = df_this.loc[(df_this["date"] >= ct2_start_calc) & (df_this["date"] <= ct2_end_calc)].reset_index(drop=True) if not df_this.empty else pd.DataFrame()

        return jsonify(replace_nan_with_none({
            "average": df_avg.to_dict(orient="records"), "this_year": df_this_json.to_dict(orient="records"), "forecast": df_forecast_json.to_dict(orient="records"),
            "ct1_period": dataframe_to_records_with_iso_date(ct1_period_df), "ct1": dataframe_to_records_with_iso_date(df_ct1), "gdd1_target": closest1,
            "ct2_period": dataframe_to_records_with_iso_date(ct2_period_df), "ct2": dataframe_to_records_with_iso_date(df_ct2), "gdd2_target": closest2,
            "ct1_until_yesterday": hist_dict1, "ct2_until_yesterday": hist_dict2
        }))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
