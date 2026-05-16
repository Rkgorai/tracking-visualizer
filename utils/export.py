import json
import csv
from pathlib import Path


def export_mot_format(tracks_per_frame: list, output_path: str) -> None:
    """
    Export tracking results in MOTChallenge format.

    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    with open(output_path, "w") as f:
        for frame_idx, tracks in enumerate(tracks_per_frame, start=1):
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                w, h = x2 - x1, y2 - y1
                # Format: frame, id, x, y, w, h, conf, -1, -1, -1
                line = f"{frame_idx},{track.track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{track.score:.4f},-1,-1,-1\n"
                f.write(line)


def export_csv(tracks_per_frame: list, output_path: str) -> None:
    """Export tracking results as CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "score", "class_id", "class_name"])

        for frame_idx, tracks in enumerate(tracks_per_frame, start=1):
            for track in tracks:
                writer.writerow([
                    frame_idx,
                    track.track_id,
                    track.bbox[0],
                    track.bbox[1],
                    track.bbox[2],
                    track.bbox[3],
                    track.score,
                    track.class_id,
                    track.class_name
                ])


def export_json(tracks_per_frame: list, output_path: str) -> None:
    """Export tracking results as JSON."""
    results = []

    for frame_idx, tracks in enumerate(tracks_per_frame, start=1):
        frame_data = {
            "frame": frame_idx,
            "tracks": []
        }

        for track in tracks:
            track_data = {
                "track_id": track.track_id,
                "bbox": track.bbox.tolist() if hasattr(track.bbox, "tolist") else list(track.bbox),
                "score": track.score,
                "class_id": track.class_id,
                "class_name": track.class_name,
                "trajectory": track.trajectory
            }
            frame_data["tracks"].append(track_data)

        results.append(frame_data)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def export_metrics(metrics_dict: dict, output_path: str, format: str = "json") -> None:
    """Export metrics in specified format."""
    if format == "json":
        with open(output_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
    elif format == "csv":
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    writer.writerow([key, value])
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_mot_format(file_path: str) -> list:
    """Load tracking results from MOT format file."""
    tracks_per_frame = {}

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue

            frame = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])

            bbox = [x, y, x + w, y + h]

            if frame not in tracks_per_frame:
                tracks_per_frame[frame] = []

            tracks_per_frame[frame].append({
                "track_id": track_id,
                "bbox": bbox,
                "score": conf
            })

    # Convert to list
    max_frame = max(tracks_per_frame.keys()) if tracks_per_frame else 0
    result = [tracks_per_frame.get(i, []) for i in range(1, max_frame + 1)]

    return result