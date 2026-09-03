import math

# Standard simplified oil-spill drift model:
# drift_velocity = (wind_drift_factor * wind_velocity) + current_velocity
# wind_drift_factor is commonly ~0.03 (3%) of wind speed, angled slightly off
# wind direction due to Coriolis effect (Northern Hemisphere: deflect right ~15-25deg)
# This is a widely used simplified approximation (not a full physics simulation).

WIND_DRIFT_FACTOR = 0.03
CORIOLIS_DEFLECTION_DEG = 20  # Northern Hemisphere approximation


def estimate_drift(wind_speed_ms, wind_dir_deg, current_speed_ms, current_dir_deg,
                    hemisphere="N"):
    """
    wind_dir_deg / current_dir_deg: direction the wind/current is GOING TO (met/ocean
        convention varies - be careful with your data source's convention).
    Returns drift speed (m/s) and direction (deg) the slick is expected to move.
    """
    deflection = CORIOLIS_DEFLECTION_DEG if hemisphere == "N" else -CORIOLIS_DEFLECTION_DEG
    wind_drift_dir = wind_dir_deg + deflection
    wind_drift_speed = wind_speed_ms * WIND_DRIFT_FACTOR

    # Convert both vectors to x/y components, add them, convert back
    def to_xy(speed, direction_deg):
        rad = math.radians(direction_deg)
        return speed * math.sin(rad), speed * math.cos(rad)

    wx, wy = to_xy(wind_drift_speed, wind_drift_dir)
    cx, cy = to_xy(current_speed_ms, current_dir_deg)
    total_x, total_y = wx + cx, wy + cy

    drift_speed = math.hypot(total_x, total_y)
    drift_dir = math.degrees(math.atan2(total_x, total_y)) % 360

    return drift_speed, drift_dir


def move_point(lat, lon, speed_ms, direction_deg, hours):
    """
    Move a lat/lon point by a given speed/direction over a time period.
    Simple flat-earth approximation - fine for short distances (hours to a
    couple days of drift), not for very long distances.
    """
    distance_m = speed_ms * hours * 3600
    rad = math.radians(direction_deg)
    dx_m = distance_m * math.sin(rad)
    dy_m = distance_m * math.cos(rad)

    # ~111,320 m per degree latitude; longitude scales by cos(latitude)
    dlat = dy_m / 111_320
    dlon = dx_m / (111_320 * math.cos(math.radians(lat)))

    return lat + dlat, lon + dlon


def backtrack_origin(observed_lat, observed_lon, wind_speed_ms, wind_dir_deg,
                      current_speed_ms, current_dir_deg, hours_since_release,
                      hemisphere="N"):
    """
    Given where a slick was OBSERVED, estimate where it likely STARTED by
    reversing the drift vector over the elapsed time.
    """
    drift_speed, drift_dir = estimate_drift(
        wind_speed_ms, wind_dir_deg, current_speed_ms, current_dir_deg, hemisphere
    )
    reverse_dir = (drift_dir + 180) % 360
    origin_lat, origin_lon = move_point(
        observed_lat, observed_lon, drift_speed, reverse_dir, hours_since_release
    )
    return {
        "estimated_origin": (origin_lat, origin_lon),
        "drift_speed_ms": drift_speed,
        "drift_direction_deg": drift_dir,
        "total_distance_traveled_km": drift_speed * hours_since_release * 3600 / 1000,
    }


if __name__ == "__main__":
    # Example only - swap in real wind/current values for a real case
    # (e.g. from ERA5 / Copernicus Marine for one of the 3 historical incidents)
    result = backtrack_origin(
        observed_lat=29.65, observed_lon=-94.98,   # example: near Houston Ship Channel
        wind_speed_ms=6.0, wind_dir_deg=180,        # wind blowing toward the north
        current_speed_ms=0.3, current_dir_deg=90,   # current flowing toward the east
        hours_since_release=12,
    )
    print("Drift estimate:")
    for k, v in result.items():
        print(f"  {k}: {v}")