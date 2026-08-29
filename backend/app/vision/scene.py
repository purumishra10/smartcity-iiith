"""Local still-content description using YOLOv8n (COCO). CPU, no cloud APIs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import settings

# COCO ids we care about for civic stills
_PERSON = 0
_BICYCLE = 1
_CAR = 2
_MOTORCYCLE = 3
_BUS = 5
_TRUCK = 7
_TRAFFIC_LIGHT = 9
_STOP_SIGN = 11

_VEHICLE_IDS = {_CAR, _MOTORCYCLE, _BUS, _TRUCK}

_MODEL = None
_MODEL_ERROR: str | None = None


def _load_model():
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None or _MODEL_ERROR:
        return _MODEL
    try:
        import shutil

        from ultralytics import YOLO

        path = Path(settings.yolo_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            _MODEL = YOLO(str(path))
        else:
            _MODEL = YOLO("yolov8n.pt")
            downloaded = Path("yolov8n.pt")
            if downloaded.exists() and downloaded.resolve() != path.resolve():
                shutil.copy(downloaded, path)
        _MODEL_ERROR = None
    except Exception as exc:  # noqa: BLE001
        _MODEL_ERROR = str(exc)
        _MODEL = None
    return _MODEL


def _count(boxes, cls_ids: set[int], conf_min: float = 0.28) -> int:
    n = 0
    if boxes is None or boxes.cls is None:
        return 0
    clses = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(clses))
    for c, p in zip(clses, confs):
        if int(c) in cls_ids and float(p) >= conf_min:
            n += 1
    return n


def _phrase(n: int, singular: str, plural: str | None = None) -> str | None:
    if n <= 0:
        return None
    if n == 1:
        return f"1 {singular}"
    return f"{n} {plural or singular + 's'}"


def describe_scene(bgr: np.ndarray, context: str | None) -> dict:
    ctx = (context or "").lower()
    source = {
        "street": "a street-level photograph",
        "camera": "a CCTV or roadside camera view",
        "other": "an uploaded civic still",
    }.get(ctx, "an uploaded still")

    model = _load_model()
    people = vehicles = bicycles = lights = signs = 0
    cars = motorcycles = buses = trucks = 0

    if model is not None:
        result = model.predict(source=bgr, verbose=False, device="cpu", imgsz=640, conf=0.28)[0]
        boxes = result.boxes
        people = _count(boxes, {_PERSON})
        cars = _count(boxes, {_CAR})
        motorcycles = _count(boxes, {_MOTORCYCLE})
        buses = _count(boxes, {_BUS})
        trucks = _count(boxes, {_TRUCK})
        vehicles = cars + motorcycles + buses + trucks
        bicycles = _count(boxes, {_BICYCLE})
        lights = _count(boxes, {_TRAFFIC_LIGHT})
        signs = _count(boxes, {_STOP_SIGN})

    bits = [f"This looks like {source}"]
    objects: list[str] = []
    for phrase in (
        _phrase(people, "pedestrian", "pedestrians"),
        _phrase(cars, "car"),
        _phrase(motorcycles, "motorcycle"),
        _phrase(buses, "bus", "buses"),
        _phrase(trucks, "truck"),
        _phrase(bicycles, "bicycle"),
        _phrase(lights, "traffic light"),
        _phrase(signs, "stop sign"),
    ):
        if phrase:
            objects.append(phrase)

    if objects:
        content = "Visible in the frame: " + ", ".join(objects) + "."
    elif model is None:
        content = (
            "Object detector is not loaded, so cars and people could not be counted. "
            "Inspect the photo on the left."
        )
    else:
        content = (
            "No cars, buses, bikes, or pedestrians were detected at the usual confidence "
            "threshold (distant or heavily occluded figures can be missed)."
        )

    if vehicles >= 4:
        traffic = "Traffic is present — several vehicles share the roadway."
    elif vehicles >= 1:
        traffic = "At least one vehicle is visible."
    elif people >= 1:
        traffic = "People are visible; the roadway may be quiet or parked-up."
    else:
        traffic = "No clear traffic or pedestrians were detected."

    scene = f"{bits[0]}. {content} {traffic}"
    return {
        "people": people,
        "vehicles": vehicles,
        "cars": cars,
        "buses": buses,
        "trucks": trucks,
        "motorcycles": motorcycles,
        "bicycles": bicycles,
        "traffic_lights": lights,
        "stop_signs": signs,
        "detector": "yolov8n" if model is not None else "unavailable",
        "appearance": scene,
        "maps": content,
        "usefulness": traffic,
        "full": scene,
    }
