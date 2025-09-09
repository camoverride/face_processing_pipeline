import platform
import cv2
import numpy as np



# Globals for tracking capture type.
_cap = None
_picam2 = None


def _init_camera(config: dict):
    """
    Detect system and initialize the camera only once.
    """
    global _cap, _picam2

    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read().lower()
                is_pi = "raspberry pi" in cpuinfo
        except FileNotFoundError:
            is_pi = False

        if is_pi:
            from picamera2 import Picamera2  # type: ignore

            _picam2 = Picamera2()

            # Get the max sensor resolution: (width, height)
            max_resolution = _picam2.sensor_resolution
            # max_resolution = _picam2.camera_properties.get("PixelArraySize", (640, 480))  # type: ignore

            _picam2.configure(
                _picam2.create_preview_configuration(
                    main={
                        "format": "RGB888",
                        "size": (max_resolution
                        ),
                    }
                )
            )
            _picam2.start()
            print(f"Raspberry Pi detected — using PiCam at resolution {max_resolution}.")

        else:
            _cap = cv2.VideoCapture(0)
            if not _cap.isOpened():
                raise RuntimeError("Error: Could not open webcam.")
            print("Linux (non-Pi) detected — using webcam.")

    elif system == "Darwin":
        _cap = cv2.VideoCapture(0)
        if not _cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        print("macOS detected — using laptop webcam.")

    else:
        raise NotImplementedError(f"Unsupported system: {system}")


def get_image(config: dict) -> np.ndarray:
    """
    Capture a single frame from the initialized camera.
    Must call _init_camera(config) first (done automatically on first use).
    """
    global _cap, _picam2

    if _cap is None and _picam2 is None:
        _init_camera(config)

    if _picam2 is not None:
        return _picam2.capture_array()

    if _cap is not None:
        ret, frame = _cap.read()
        if not ret:
            raise RuntimeError("Error: Failed to capture image from webcam.")
        return frame

    raise RuntimeError("No camera initialized.")
