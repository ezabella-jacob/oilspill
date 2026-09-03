import pandas as pd
import numpy as np
import math


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def load_ais_csv(path):
    """
    Expects NOAA MarineCadastre AIS format columns:
    MMSI, BaseDateTime, LAT, LON, SOG (speed over ground, knots),
    COG (course over ground, deg), Heading, VesselName, VesselType
    """
    df = pd.read_csv(path, parse_dates=["BaseDateTime"])
    return df


def find_candidate_vessels(ais_df, origin_lat, origin_lon, event_time,
                            time_window_hours=24, search_radius_km=50):
    """
    Filters AIS records to vessels present near the estimated spill origin
    within a time window, then computes per-vessel features for ranking.
    """
    window_start = event_time - pd.Timedelta(hours=time_window_hours)
    window_end = event_time + pd.Timedelta(hours=time_window_hours)

    nearby = ais_df[
        (ais_df["BaseDateTime"] >= window_start) &
        (ais_df["BaseDateTime"] <= window_end)
    ].copy()

    if nearby.empty:
        return pd.DataFrame()

    nearby["distance_km"] = nearby.apply(
        lambda r: haversine_km(origin_lat, origin_lon, r["LAT"], r["LON"]), axis=1
    )
    nearby = nearby[nearby["distance_km"] <= search_radius_km]

    if nearby.empty:
        return pd.DataFrame()

    results = []
    for mmsi, group in nearby.groupby("MMSI"):
        closest = group.loc[group["distance_km"].idxmin()]
        time_diff_hours = abs((closest["BaseDateTime"] - event_time).total_seconds()) / 3600

        results.append({
            "MMSI": mmsi,
            "VesselName": closest.get("VesselName", "Unknown"),
            "VesselType": closest.get("VesselType", "Unknown"),
            "closest_distance_km": closest["distance_km"],
            "time_diff_hours": time_diff_hours,
            "speed_at_closest_knots": closest.get("SOG", np.nan),
            "course_at_closest_deg": closest.get("COG", np.nan),
            "num_positions_in_window": len(group),
        })

    return pd.DataFrame(results).sort_values("closest_distance_km").reset_index(drop=True)


def generate_fake_ais_data(origin_lat, origin_lon, event_time, n_vessels=5, seed=0):
    """For testing without real downloaded AIS data."""
    rng = np.random.default_rng(seed)
    rows = []
    for mmsi in range(100000000, 100000000 + n_vessels):
        base_lat = origin_lat + rng.uniform(-0.5, 0.5)
        base_lon = origin_lon + rng.uniform(-0.5, 0.5)
        for hour_offset in range(-12, 13, 2):
            t = event_time + pd.Timedelta(hours=hour_offset)
            rows.append({
                "MMSI": mmsi,
                "BaseDateTime": t,
                "LAT": base_lat + rng.uniform(-0.05, 0.05),
                "LON": base_lon + rng.uniform(-0.05, 0.05),
                "SOG": rng.uniform(0, 15),
                "COG": rng.uniform(0, 360),
                "VesselName": f"TESTVESSEL{mmsi}",
                "VesselType": "Cargo",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    event_time = pd.Timestamp("2019-05-10 12:22:54")
    origin_lat, origin_lon = 29.68, -94.98  # example, e.g. from drift_model.py output

    fake_ais = generate_fake_ais_data(origin_lat, origin_lon, event_time)
    candidates = find_candidate_vessels(fake_ais, origin_lat, origin_lon, event_time)

    print("Candidate vessels near estimated origin:")
    print(candidates.to_string(index=False))