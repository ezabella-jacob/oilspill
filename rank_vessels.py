import numpy as np
import pandas as pd


def score_vessels(candidates_df, estimated_drift_dir_deg,
                   max_distance_km=50, max_time_hours=24):
    """
    Takes the candidate vessel table from ais_correlation.py and adds a
    composite score (0-1) per vessel, plus a plain-text explanation.

    Score combines three signals, each normalized to 0-1 and weighted:
      - proximity: closer to origin = higher score (linear decay to 0 at max_distance_km)
      - time: closer in time to the event = higher score (linear decay to 0 at max_time_hours)
      - course_consistency: is the vessel's course roughly aligned with the
        estimated drift direction? (a vessel moving the same way oil would
        drift is a weaker "smoking gun" than one moving the opposite way,
        but this is a simple heuristic, not proof)

    Weights are deliberately simple/transparent rather than learned, so each
    score is explainable - important for a system meant to support human
    judgment, not replace it.
    """
    df = candidates_df.copy()

    W_PROXIMITY = 0.45
    W_TIME = 0.30
    W_COURSE = 0.25

    proximity_score = np.clip(1 - (df["closest_distance_km"] / max_distance_km), 0, 1)
    time_score = np.clip(1 - (df["time_diff_hours"] / max_time_hours), 0, 1)

    def course_alignment_score(course_deg):
        if pd.isna(course_deg):
            return 0.5  # unknown - neutral score, don't penalize or reward
        diff = abs((course_deg - estimated_drift_dir_deg + 180) % 360 - 180)
        return 1 - (diff / 180)  # 1.0 = same direction, 0.0 = opposite direction

    course_score = df["course_at_closest_deg"].apply(course_alignment_score)

    df["proximity_score"] = proximity_score.round(3)
    df["time_score"] = time_score.round(3)
    df["course_score"] = course_score.round(3)
    df["total_score"] = (
        W_PROXIMITY * proximity_score +
        W_TIME * time_score +
        W_COURSE * course_score
    ).round(3)

    def explain(row):
        parts = []
        parts.append(f"{row['closest_distance_km']:.1f}km from origin")
        parts.append(f"{row['time_diff_hours']:.1f}h from event time")
        if row["course_score"] > 0.7:
            parts.append("course aligns with estimated drift")
        elif row["course_score"] < 0.3:
            parts.append("course opposes estimated drift")
        return ", ".join(parts)

    df["explanation"] = df.apply(explain, axis=1)

    return df.sort_values("total_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from ais_correlation import generate_fake_ais_data, find_candidate_vessels
    from drift_model import estimate_drift

    event_time = pd.Timestamp("2019-05-10 12:22:54")
    origin_lat, origin_lon = 29.68, -94.98

    fake_ais = generate_fake_ais_data(origin_lat, origin_lon, event_time)
    candidates = find_candidate_vessels(fake_ais, origin_lat, origin_lon, event_time)

    _, drift_dir = estimate_drift(
        wind_speed_ms=6.0, wind_dir_deg=180,
        current_speed_ms=0.3, current_dir_deg=90,
    )

    ranked = score_vessels(candidates, estimated_drift_dir_deg=drift_dir)

    print(f"Estimated drift direction: {drift_dir:.1f} deg\n")
    print("Ranked vessel candidates:\n")
    for i, row in ranked.iterrows():
        print(f"{i+1}. {row['VesselName']} (MMSI {row['MMSI']}) - score {row['total_score']}")
        print(f"   {row['explanation']}")
        print()