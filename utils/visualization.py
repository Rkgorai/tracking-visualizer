import numpy as np
import cv2


# YOLO-style color palette - distinct colors for different tracks
YOLO_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Pink
]


def get_color(track_id: int) -> tuple:
    """Get YOLO-style color for a track ID."""
    return YOLO_COLORS[track_id % len(YOLO_COLORS)]


def draw_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    track_id: int,
    score: float = 1.0,
    class_name: str = "",
    thickness: int = 3,
    show_id: bool = True,
    show_score: bool = True
) -> np.ndarray:
    """Draw YOLO-style bounding box with track ID."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame
    x1, y1, x2, y2 = map(int, bbox)
    color = get_color(track_id)

    # Draw filled rectangle with border (YOLO style)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw corner highlights (YOLO style - thicker at corners)
    corner_len = 15
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

    # Create label in YOLO style: "track:1 0.95" or "person:1 0.95"
    if class_name:
        label = f"{class_name}:{track_id} {score:.2f}"
    else:
        label = f"track:{track_id} {score:.2f}"

    # Draw label background with padding
    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    label_y = y1 - label_h - 8

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1 - 2, label_y - 4),
        (x1 + label_w + 4, y1 + 2),
        color,
        -1
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text
    cv2.putText(
        frame, label, (x1 + 2, label_y + label_h),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    return frame


def draw_trajectory(
    frame: np.ndarray,
    trajectory: list,
    track_id: int,
    max_length: int = 50,
    thickness: int = 2
) -> np.ndarray:
    """Draw trajectory trail."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame
    if len(trajectory) < 2:
        return frame

    color = get_color(track_id)
    points = trajectory[-max_length:]

    for i in range(1, len(points)):
        pt1 = (int(points[i-1][0]), int(points[i-1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))
        cv2.line(frame, pt1, pt2, color, thickness)

    return frame


def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    show_trajectory: bool = True,
    max_trail_length: int = 30
) -> np.ndarray:
    """Draw all tracks on a frame."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame
    for track in tracks:
        frame = draw_bbox(
            frame,
            track.bbox,
            track.track_id,
            track.score,
            track.class_name
        )

        if show_trajectory and len(track.trajectory) > 1:
            frame = draw_trajectory(
                frame,
                track.trajectory,
                track.track_id,
                max_length=max_trail_length
            )

    return frame


def draw_fps(frame: np.ndarray, fps: float, position: tuple = (10, 30)) -> np.ndarray:
    """Draw FPS counter on frame."""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    return frame


def draw_metrics_overlay(
    frame: np.ndarray,
    metrics: dict,
    position: tuple = (10, 60)
) -> np.ndarray:
    """Draw metrics overlay on frame."""
    y_offset = position[1]
    for key, value in metrics.items():
        text = f"{key}: {value}"
        cv2.putText(
            frame, text, (position[0], y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        y_offset += 20
    return frame


def create_grid_visualization(
    frames: list,
    titles: list,
    grid_size: tuple = (1, 1)
) -> np.ndarray:
    """Create a grid visualization of multiple frames."""
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    h, w = frames[0].shape[:2]
    rows, cols = grid_size

    # Resize all frames to same size
    resized = [cv2.resize(f, (w, h)) for f in frames]

    # Create grid
    grid_rows = []
    for i in range(rows):
        row_frames = resized[i * cols:(i + 1) * cols]
        # Add titles
        for j, (f, title) in enumerate(zip(row_frames, titles[i * cols:(i + 1) * cols])):
            cv2.putText(f, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        grid_rows.append(np.hstack(row_frames))

    return np.vstack(grid_rows)


def save_annotated_video(
    input_path: str,
    output_path: str,
    tracks_per_frame: list,
    fps: float = 30.0
) -> None:
    """Save video with annotations."""
    import cv2

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(tracks_per_frame):
            frame = draw_tracks(frame, tracks_per_frame[frame_idx])

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()