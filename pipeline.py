import numpy as np
import torch
import pandas as pd
import segmentation_models_pytorch as smp

from dataset import OilSpillDataset
from characterize_spill import characterize_spill
from drift_model import backtrack_origin
from ais_correlation import generate_fake_ais_data, find_candidate_vessels
from rank_vessels import score_vessels


def pixel_to_latlon(pixel_x, pixel_y, geotransform):
    """
    Standard GDAL-style affine geotransform:
    geotransform = (origin_lon, pixel_width_deg, 0, origin_lat, 0, -pixel_height_deg)
    This is FAKE for now (no real georeferenced scene wired up yet) - but the
    math here is exactly what you'd use with a real GeoTIFF's real geotransform,
    which you can read directly from a real Sentinel-1 product with rasterio/GDAL.
    """
    origin_lon, pixel_width, _, origin_lat, _, neg_pixel_height = geotransform
    lon = origin_lon + pixel_x * pixel_width
    lat = origin_lat + pixel_y * neg_pixel_height
    return lat, lon


def run_pipeline(image_tensor, event_time, geotransform,
                  wind_speed_ms, wind_dir_deg, current_speed_ms, current_dir_deg,
                  hours_since_release, device):
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1).to(device)
    model.load_state_dict(torch.load("best_unet.pth", map_location=device))
    model.eval()

    # --- Stage 1: detect ---
    with torch.no_grad():
        pred = torch.sigmoid(model(image_tensor.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
    mask_binary = (pred > 0.5).astype(np.uint8)

    # --- Stage 2: characterize (pixel space) ---
    stats = characterize_spill(mask_binary, pixel_size_m=10.0)
    if not stats["found"]:
        print("No spill detected.")
        return

    # --- Bridge: pixel centroid -> real-world lat/lon (FAKE geotransform for now) ---
    cx_px, cy_px = stats["centroid_px"]
    origin_lat, origin_lon = pixel_to_latlon(cx_px, cy_px, geotransform)
    print(f"Detected spill centroid: pixel ({cx_px:.1f}, {cy_px:.1f}) -> lat/lon ({origin_lat:.5f}, {origin_lon:.5f})")
    print(f"Estimated area: {stats['area_km2']:.3f} km2\n")

    # --- Stage 3: drift / backtrack ---
    drift_result = backtrack_origin(
        origin_lat, origin_lon, wind_speed_ms, wind_dir_deg,
        current_speed_ms, current_dir_deg, hours_since_release,
    )
    est_origin_lat, est_origin_lon = drift_result["estimated_origin"]
    print(f"Estimated release origin (backtracked): ({est_origin_lat:.5f}, {est_origin_lon:.5f})")
    print(f"Drift direction: {drift_result['drift_direction_deg']:.1f} deg, "
          f"distance traveled: {drift_result['total_distance_traveled_km']:.2f} km\n")

    # --- Stage 4: AIS correlation (fake data for now - swap for real CSV later) ---
    fake_ais = generate_fake_ais_data(est_origin_lat, est_origin_lon, event_time)
    candidates = find_candidate_vessels(fake_ais, est_origin_lat, est_origin_lon, event_time)
    if candidates.empty:
        print("No candidate vessels found in range.")
        return

    # --- Stage 5: rank ---
    ranked = score_vessels(candidates, estimated_drift_dir_deg=drift_result["drift_direction_deg"])

    print("Ranked candidate vessels:\n")
    for i, row in ranked.iterrows():
        print(f"{i+1}. {row['VesselName']} (MMSI {row['MMSI']}) - score {row['total_score']}")
        print(f"   {row['explanation']}")
        print()

    return {
        "spill_detected": True,
        "area_km2": stats["area_km2"],
        "centroid_latlon": [origin_lat, origin_lon],
        "estimated_origin_latlon": [est_origin_lat, est_origin_lon],
        "drift_direction_deg": drift_result["drift_direction_deg"],
        "distance_traveled_km": drift_result["total_distance_traveled_km"],
        "ranked_vessels": ranked.to_dict(orient="records"),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = OilSpillDataset(
        r"C:\Users\ezabe\Downloads\dataset\test\sentinel\image",
        r"C:\Users\ezabe\Downloads\dataset\test\sentinel\label",
        patch_size=256,
        augment=False,
    )
    img, _ = test_ds[0]  # the S-shaped slick sample from earlier - our best case

    # FAKE geotransform - pretends this 256x256 patch sits near Galveston Bay,
    # TX, with ~10m pixels (roughly Sentinel-1 GRD resolution). Replace with a
    # real geotransform once using an actual downloaded Sentinel-1 scene.
    fake_geotransform = (-94.99, 0.0001, 0, 29.69, 0, -0.0001)

    run_pipeline(
        image_tensor=img,
        event_time=pd.Timestamp("2019-05-10 12:22:54"),
        geotransform=fake_geotransform,
        wind_speed_ms=6.0, wind_dir_deg=180,
        current_speed_ms=0.3, current_dir_deg=90,
        hours_since_release=12,
        device=device,
    )