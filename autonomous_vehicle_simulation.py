# -*- coding: utf-8 -*-
"""
Integrated Vehicle Sensor Simulation
This module provides a comprehensive system for configuring and simulating vehicle sensors
with complete 3D positioning and orientation.
"""
import math
import numpy as np
import random
import uuid
from typing import Dict, List, Tuple, Any, Optional, Set
import matplotlib.pyplot as plt

class Position3D:
    """
    Represents a 3D position and orientation in space.

    Attributes:
        x (float): X coordinate (forward/backward)
        y (float): Y coordinate (left/right)
        z (float): Z coordinate (up/down)
        roll (float): Roll angle in degrees (rotation around X-axis)
        pitch (float): Pitch angle in degrees (rotation around Y-axis)
        yaw (float): Yaw angle in degrees (rotation around Z-axis)

    >>> p1 = Position3D(0, 0, 0)
    >>> p2 = Position3D(3, 4, 0)
    >>> p1.distance_to(p2)
    5.0
    >>> tuple(round(x, 2) for x in p1.direction_to(p2))
    (0.0, 0.0, 53.13)
    >>> p1 = Position3D(0, 0, 0, 0, 0, 0)
    >>> p2 = Position3D(10, 0, 0)
    >>> p1.is_within_field_of_view(p2, 120, 90)
    True
    """
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll  # in degrees
        self.pitch = pitch  # in degrees
        self.yaw = yaw      # in degrees

    def __str__(self) -> str:
        return f"Position3D(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, " \
               f"roll={self.roll:.2f}°, pitch={self.pitch:.2f}°, yaw={self.yaw:.2f}°)"

    def to_dict(self) -> Dict[str, float]:
        """Convert position to dictionary format."""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position3D':
        """Create Position3D object from dictionary."""
        return cls(
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            z=data.get('z', 0.0),
            roll=data.get('roll', 0.0),
            pitch=data.get('pitch', 0.0),
            yaw=data.get('yaw', 0.0)
        )

    def distance_to(self, other: 'Position3D') -> float:
        """ Calculate Euclidean distance to another Position3D."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def direction_to(self, other: 'Position3D') -> Tuple[float, float, float]:
        """
        Calculate direction (roll, pitch, yaw) from this position to another,
        taking into account the current orientation.

        Returns:
            tuple: (roll, pitch, yaw) in degrees
        """
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z

        # Calculate horizontal distance
        horizontal_distance = math.sqrt(dx**2 + dy**2)

        # Calculate pitch (elevation angle)
        target_pitch = math.degrees(math.atan2(dz, horizontal_distance))

        # Calculate yaw (azimuth angle)
        target_yaw = math.degrees(math.atan2(dy, dx))

        # Calculate roll - in this simple scenario, using the same roll as the current position
        target_roll = self.roll

        # Relative pitch (considering current pitch orientation)
        relative_pitch = target_pitch - self.pitch

        # Adjusting yaw for current orientation - normalizing to -180 to +180 range
        relative_yaw = target_yaw - self.yaw
        if relative_yaw > 180:
            relative_yaw -= 360
        elif relative_yaw < -180:
            relative_yaw += 360

        # Calculate relative roll
        relative_roll = target_roll - self.roll
        # Normalize to -180 to +180 range
        if relative_roll > 180:
            relative_roll -= 360
        elif relative_roll < -180:
            relative_roll += 360

        return relative_roll, relative_pitch, relative_yaw

    def is_within_field_of_view(self, target: 'Position3D',
                               horizontal_fov: float, vertical_fov: float) -> bool:
        """
        Check if target position is within field of view from this position.

        Args:
            target: Target position to check
            horizontal_fov: Horizontal field of view in degrees
            vertical_fov: Vertical field of view in degrees

        Returns:
            bool: True if target is within field of view
        """
        _, rel_pitch, rel_yaw = self.direction_to(target)

        return (abs(rel_yaw)   <= horizontal_fov / 2 and
                abs(rel_pitch) <= vertical_fov   / 2)

class Sensor:
    """
    Base class for vehicle sensors.

    Attributes:
        name (str): Sensor name
        type (str): Sensor type (camera, lidar, radar, etc.)
        position (Position3D): Sensor's position and orientation
        range (float): Maximum detection range in meters
        id (str): Unique identifier for the sensor

    >>> pos = Position3D(0, 0, 0)
    >>> target = Position3D(0, 30, 0)
    >>> sensor = Sensor("GenericSensor", "custom", pos, range=50)
    >>> sensor.can_detect(target)
    True

    >>> target_far = Position3D(0, 100, 0)
    >>> sensor.can_detect(target_far)
    False

    >>> str(sensor)  # doctest: +ELLIPSIS
    " 'GenericSensor' at Position3D(x=0.00, y=0.00, z=0.00, roll=0.00°, pitch=0.00°, yaw=0.00°) "
    """
    def __init__(self, name: str, sensor_type: str, position: Position3D, range: float):
        self.name = name
        self.type = sensor_type
        self.position = position
        self.range = range
        self.id = str(uuid.uuid4())  # Assign a unique ID

    def __str__(self) -> str:
        return f" '{self.name}' at {self.position} "

    def can_detect(self, target_position: Position3D) -> bool:
        """
        Check if a target is within detection range.

        Args:
            target_position: Position of target to detect

        Returns:
            bool: True if target is within range
        """
        distance = self.position.distance_to(target_position)
        return distance <= self.range


class Camera(Sensor):
    """
    Camera sensor for visual detection.

    Attributes:
        resolution (Tuple[int, int]): Camera resolution (width, height)
        fov (float): Horizontal field of view in degrees

    >>> cam_pos = Position3D(0, 0, 0, 0, 0, 0)
    >>> cam = Camera("FrontCam", cam_pos, range=100, resolution=(1920, 1080), fov=90)
    >>> target = Position3D(10, 0, 0)
    >>> cam.can_detect(target)
    True
    """
    def __init__(self, name: str, position: Position3D, range: float,
                 resolution: Tuple[int, int], fov: float):
        super().__init__(name, "camera", position, range)
        self.resolution = resolution
        self.horizontal_fov = fov
        self.aspect_ratio = (resolution[1] / resolution[0])
        self.vertical_fov = 2 * math.atan(math.tan(math.radians(fov / 2)) / self.aspect_ratio)

    def can_detect(self, target_position: Position3D) -> bool:
        """
        Check if target is within camera's field of view and range.

        Args:
            target_position: Position of target to detect

        Returns:
            bool: True if target is within range and FOV
        """
        if not super().can_detect(target_position):
            return False

        # Check if within field of view
        return self.position.is_within_field_of_view(
            target_position, self.horizontal_fov, self.vertical_fov
        )


class Lidar(Sensor):
    """
    Lidar sensor for 3D point cloud detection.

    Attributes:
        horizontal_fov (float): Horizontal field of view in degrees
        vertical_fov (float): Vertical field of view in degrees

    >>> cam_pos = Position3D(0, 0, 0, 0, 0, 0)
    >>> target = Position3D(10, 0, 0)
    >>> lidar = Lidar("LidarTop", cam_pos, range=100, horizontal_fov=360, vertical_fov=60)
    >>> lidar.can_detect(target)
    True
    """
    def __init__(self, name: str, position: Position3D, range: float,
                 horizontal_fov: float, vertical_fov: float):
        super().__init__(name, "lidar", position, range)
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

    def can_detect(self, target_position: Position3D) -> bool:
        """
        Check if target is within lidar's field of view and range.

        Args:
            target_position: Position of target to detect

        Returns:
            bool: True if target is within range and FOV
        """
        if not super().can_detect(target_position):
            return False

        # Check if within field of view
        return self.position.is_within_field_of_view(
            target_position, self.horizontal_fov, self.vertical_fov
        )


class Radar(Sensor):
    """
    Radar sensor for velocity and distance detection.

    Attributes:
        fov (float): Field of view in degrees
        resolution (str): Resolution quality (low, medium, high)

    >>> cam_pos = Position3D(0, 0, 0, 0, 0, 0)
    >>> target = Position3D(10, 0, 0)
    >>> radar = Radar("RadarFront", cam_pos, range=100, fov=120)
    >>> radar.can_detect(target)
    True
    """
    def __init__(self, name: str, position: Position3D, range: float,
                 fov: float):
        super().__init__(name, "radar", position, range)
        self.fov = fov
        # For simplicity, assuming vertical FOV is 1/3 of horizontal
        self.vertical_fov = self.fov / 3

    def can_detect(self, target_position: Position3D) -> bool:
        """
        Check if target is within radar's field of view and range.

        Args:
            target_position: Position of target to detect

        Returns:
            bool: True if target is within range and FOV
        """
        if not super().can_detect(target_position):
            return False

        # Check if within field of view
        return self.position.is_within_field_of_view(
            target_position, self.fov, self.vertical_fov
        )


class VehicleSensorSystem:
    """
    Manages all sensors on a vehicle.

    Attributes:
        sensors (Dict[str, Sensor]): Dictionary of sensors by name

    >>> vss = VehicleSensorSystem()
    >>> cam_pos = Position3D(0, 0, 0, 0, 0, 0)
    >>> cam = Camera("FrontCam", cam_pos, range=100, resolution=(1920, 1080), fov=90)
    >>> target = Position3D(10, 0, 0)
    >>> lidar = Lidar("LidarTop", cam_pos, range=100, horizontal_fov=360, vertical_fov=30)
    >>> radar = Radar("RadarFront", cam_pos, range=100, fov=120)
    >>> vss.add_sensor(cam)
    >>> vss.add_sensor(lidar)
    >>> vss.add_sensor(radar)
    >>> sorted(vss.detect_target(target).items())  # ensures stable comparison
    [('FrontCam', True), ('LidarTop', True), ('RadarFront', True)]
    >>> config = {
    ...     "FrontCam": {
    ...         "type": "camera",
    ...         "position": [0, 0, 0],
    ...         "orientation": [0, 0, 0],
    ...         "range": 100,
    ...         "resolution": [1920, 1080],
    ...         "field_of_view": 90
    ...     }
    ... }
    >>> vss2 = VehicleSensorSystem()
    >>> vss2.from_configuration(config)
    >>> vss2.get_sensor("FrontCam").type
    'camera'
    """
    def __init__(self):
        self.sensors = {}

    def add_sensor(self, sensor: Sensor):
        """Add a sensor to the system."""
        self.sensors[sensor.name] = sensor

    def remove_sensor(self, sensor_name: str):
        """Remove a sensor from the system."""
        if sensor_name in self.sensors:
            del self.sensors[sensor_name]

    def get_sensor(self, sensor_name: str) -> Optional[Sensor]:
        """Get a sensor by name."""
        return self.sensors.get(sensor_name)

    def detect_target(self, target_position: Position3D) -> Dict[str, bool]:
        """
        Check which sensors can detect a target.

        Args:
            target_position: Position of target to detect

        Returns:
            dict: Dictionary mapping sensor names to detection status
        """
        results = {}
        for name, sensor in self.sensors.items():
            results[name] = sensor.can_detect(target_position)
        return results

    def get_detecting_sensors(self, target_position: Position3D) -> List[Sensor]:
        """
        Get list of sensors that can detect a target.

        Args:
            target_position: Position of target to detect

        Returns:
            list: List of sensors that can detect the target
        """
        return [
            sensor for sensor in self.sensors.values()
            if sensor.can_detect(target_position)
        ]

    def from_configuration(self, config: Dict[str, Dict[str, Any]]):
        """
        Build sensor system from a configuration dictionary.

        Args:
            config: Dictionary mapping sensor names to their configurations
        """
        for name, sensor_config in config.items():
            sensor_type = sensor_config["type"]

            # Create Position3D from config
            pos_data = sensor_config["position"]
            orientation = sensor_config["orientation"]
            position = Position3D(
                x=pos_data[0],
                y=pos_data[1],
                z=pos_data[2],
                roll=orientation[0],
                pitch=orientation[1],
                yaw=orientation[2]
            )

            # Create appropriate sensor type
            if sensor_type == "camera":
                sensor = Camera(
                    name=name,
                    position=position,
                    range=sensor_config["range"],
                    resolution=tuple(sensor_config["resolution"]),
                    fov=sensor_config["field_of_view"]
                )
            elif sensor_type == "lidar":
                sensor = Lidar(
                    name=name,
                    position=position,
                    range=sensor_config["range"],
                    horizontal_fov=sensor_config["horizontal_fov"],
                    vertical_fov=sensor_config["vertical_fov"]
                )
            elif sensor_type == "radar":
                sensor = Radar(
                    name=name,
                    position=position,
                    range=sensor_config["range"],
                    fov=sensor_config["field_of_view"]
                )
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")

            self.add_sensor(sensor)


class SensorConfigFactory:
    """
    Factory class for creating different sensor configurations.
    """
    @staticmethod
    def create_tesla_vision_config():
        """
        Creates a Tesla Vision sensor configuration (Model 3/Y) with camera-only setup.

        Returns:
            dict: A dictionary containing Tesla Vision sensor configurations
        """
        # Tesla Model 3/Y approximate dimensions (in meters)
        vehicle_length = 4.7
        vehicle_width = 1.9
        vehicle_height = 1.5

        # Reference coordinate system:
        # - Origin (0,0,0) is at the center of the vehicle projected on to the ground
        # - X-axis points forward (positive = front of vehicle)
        # - Y-axis points to the right side of the vehicle (when facing forward)
        # - Z-axis points upward

        # Tesla Vision camera configuration
        config = {
            # Three forward cameras in windshield housing
            "main_forward_camera": {
                "type": "camera",
                "position": [0.8,0.0, 1.5],  # Center windshield, upper
                "orientation": [0, 0, 0],  # [roll, pitch, yaw] in degrees
                "field_of_view": 60,  # horizontal FOV in degrees
                "range": 120,  # in meters
                "resolution": [1280, 960]  # pixels
            },
            "forward_wide_camera": {
                "type": "camera",
                "position": [0.8,0.2,1.5],  # Center windshield, upper, same housing
                "orientation": [0, 0, 0],
                "field_of_view": 120,  # wider FOV
                "range": 60,
                "resolution": [1280, 960]
            },
            "forward_narrow_camera": {
                "type": "camera",
                "position": [0.8,-0.2, 1.5],  # Center windshield, upper, same housing
                "orientation": [0, 0, 0],
                "field_of_view": 35,  # narrow FOV for long range
                "range": 180,
                "resolution": [1280, 960]
            },

            # B-pillar cameras
            "left_b_pillar_camera": {
                "type": "camera",
                "position": [0.2, -0.95, 1.15],  # Left B-pillar
                "orientation": [0, 0, 90],  # facing left
                "field_of_view": 90,
                "range": 80,
                "resolution": [1280, 960]
            },
            "right_b_pillar_camera": {
                "type": "camera",
                "position": [0.2, 0.95, 1.15],  # Right B-pillar
                "orientation": [0, 0, -90],  # facing right
                "field_of_view": 90,
                "range": 80,
                "resolution": [1280, 960]
            },

            # Front fender cameras (rearward looking)
            # Tesla's actual fender cameras likely point backward at ~110° to ~135° depending on the model and year.
            "left_front_fender_camera": {
                "type": "camera",
                "position": [1.8, -0.95, 1.05],  # Left front fender
                "orientation": [0, 0, 125],  # angled backward on left side
                "field_of_view": 90,
                "range": 80,
                "resolution": [1280, 960]
            },
            "right_front_fender_camera": {
                "type": "camera",
                "position": [1.8, 0.95, 1.05],  # Right front fender
                "orientation": [0, 0, -125],  # angled backward on right side
                "field_of_view": 90,
                "range": 80,
                "resolution": [1280, 960]
            },

            # Rear camera
            "rear_camera": {
                "type": "camera",
                "position": [-2.35, 0.0, 1.15],  # Center rear, above license plate
                "orientation": [0, 0, 180],  # facing rear
                "field_of_view": 120,
                "range": 50,
                "resolution": [1280, 960]
            }
        }

        return config

    @staticmethod
    def create_mercedes_drive_pilot_config():
        """
        Creates a Mercedes-Benz Drive Pilot sensor configuration (S-Class/EQS).

        Returns:
            dict: A dictionary containing Mercedes Drive Pilot sensor configurations
        """
        # Mercedes S-Class/EQS dimensions (in meters)
        vehicle_length = 5.2
        vehicle_width = 2.1
        vehicle_height = 1.5

        # Mercedes Drive Pilot sensor configuration
        config = {
            # Camera Systems
            "stereo_multi_purpose_camera": {
                "type": "camera",
                "position": [0.9, 0.0, 1.4],  # Behind windshield, upper center
                "orientation": [0, 0, 0],  # Forward-facing
                "field_of_view": 45,  # Horizontal FOV in degrees
                "range": 150,  # in meters
                "resolution": [1920, 1080]  # pixels
            },
            "long_range_camera": {
                "type": "camera",
                "position": [0.9, 0.1, 1.4],  # Behind windshield, adjacent to stereo camera
                "orientation": [0, 0, 0],
                "field_of_view": 25,  # Narrower FOV for long range
                "range": 250,
                "resolution": [1920, 1080]
            },
            "front_surround_camera": {
                "type": "camera",
                "position": [2.55, 0.0, 0.7],  # Front grille
                "orientation": [0, 0, 0],
                "field_of_view": 180,
                "range": 20,
                "resolution": [1280, 960]
            },
            "left_surround_camera": {
                "type": "camera",
                "position": [0.0, -1.05, 0.9],  # Left side mirror
                "orientation": [0, 0, 90],  # Facing left
                "field_of_view": 180,
                "range": 20,
                "resolution": [1280, 960]
            },
            "right_surround_camera": {
                "type": "camera",
                "position": [0.0, 1.05, 0.9],  # Right side mirror
                "orientation": [0, 0, -90],  # Facing right
                "field_of_view": 180,
                "range": 20,
                "resolution": [1280, 960]
            },
            "rear_surround_camera": {
                "type": "camera",
                "position": [-2.6, 0.0, 1.0],  # Trunk/license plate area
                "orientation": [0, 0, 180],  # Facing rear
                "field_of_view": 180,
                "range": 20,
                "resolution": [1280, 960]
            },
            "night_vision_camera": {
                "type": "camera",
                "position": [2.5, 0.0, 0.5],  # Lower front grille
                "orientation": [0, 0, 0],
                "field_of_view": 24,
                "range": 160,
                "resolution": [640, 480]
            },

            # Radar Systems
            "front_long_range_radar": {
                "type": "radar",
                "position": [2.55, 0.0, 0.6],  # Front bumper center
                "orientation": [0, 0, 0],
                "field_of_view": 18,
                "range": 250,
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            },
            "front_left_corner_radar": {
                "type": "radar",
                "position": [2.4, -0.9, 0.4],  # Front left corner
                "orientation": [0, 0, 45],  # Angled outward
                "field_of_view": 150,
                "range": 80,
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            },
            "front_right_corner_radar": {
                "type": "radar",
                "position": [2.4, 0.9, 0.4],  # Front right corner
                "orientation": [0, 0, -45],  # Angled outward
                "field_of_view": 150,
                "range": 80,
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            },
            "rear_left_corner_radar": {
                "type": "radar",
                "position": [-2.4, -0.9, 0.4],  # Rear left corner
                "orientation": [0, 0, 135],  # Angled outward
                "field_of_view": 150,
                "range": 80,
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            },
            "rear_right_corner_radar": {
                "type": "radar",
                "position": [-2.4, 0.9, 0.4],  # Rear right corner
                "orientation": [0, 0, -135],  # Angled outward
                "field_of_view": 150,
                "range": 80,
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            },

            # LiDAR System (for newer models with Level 3 automation which uses a VALEO Scala 2)
            # Not a 360 degree LIDAR
            "roof_lidar": {
                "type": "lidar",
                "position": [0.7, 0.0, 1.5],  # Roof center-front
                "orientation": [0, 0, 0],
                "horizontal_fov": 145,
                "vertical_fov": 25,
                "range": 200,
                "resolution": [0.1, 0.1]  # angular resolution in degrees
            },
        }

        return config

    @staticmethod
    def create_generic_av_config():
        """
        Creates a generic autonomous vehicle sensor configuration with a typical
        sensor suite including cameras, lidars, and radars.

        Returns:
            dict: A dictionary containing sensor configurations
        """
        # Base vehicle dimensions (in meters)
        vehicle_length = 4.5  # typical mid-size vehicle length
        vehicle_width = 1.5   # typical mid-size vehicle width
        vehicle_height = 1.5  # typical mid-size vehicle height

        # Sensor configuration
        config = {
            # Cameras
            "front_camera": {
                "type": "camera",
                "position": [2.1, 0.0, 1.3],  # Center windshield, upper
                "orientation": [0, 0, 0],
                "field_of_view": 120,
                "range": 150,
                "resolution": [1920, 1080]
            },

            # Lidar
            "roof_lidar": {
                "type": "lidar",
                "position": [0.0, 0.0, 1.5],  # Center roof
                "orientation": [0, 0, 0],
                "vertical_fov": 45,
                "horizontal_fov": 145,
                "range": 150,
                "k": 0.01
            },

            # Radars
            "front_radar": {
                "type": "radar",
                "position": [2.25, 0.0, 0.4],  # Front bumper center
                "orientation": [0, 0, 0],
                "field_of_view": 60,
                "range": 200,
                "resolution": "medium",
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            },
            "rear_radar": {
                "type": "radar",
                "position": [-2.25, 0.0, 0.4],  # Rear bumper center
                "orientation": [0, 0, 180],
                "field_of_view": 60,
                "range": 150,
                "resolution": "medium",
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            },
            "front_left_radar": {
                "type": "radar",
                "position": [2.1, -0.75, 0.4],  # Front left bumper/fender
                "orientation": [0, 0, 45],
                "field_of_view": 120,
                "range": 100,
                "resolution": "medium",
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            },
            "front_right_radar": {
                "type": "radar",
                "position": [2.1, 0.75, 0.4],  # Front right bumper/fender
                "orientation": [0, 0, -45],
                "field_of_view": 120,
                "range": 100,
                "resolution": "medium",
                'a': 0.1,   # Sigmoid steepness
                'd0': 120   # Midpoint (distance where probability is ~50%)
            }
        }

        return config

    @staticmethod
    def create_hybrid_configuration():
        """
        Creates a hybrid sensor configuration by selecting sensors from Tesla Vision
        and Mercedes Drive Pilot configurations.

        For each position/function, it randomly selects either the Tesla or Mercedes sensor.

        Returns:
            dict: A dictionary containing the hybrid sensor configuration
        """
        # Get the base configurations
        tesla_config = SensorConfigFactory.create_tesla_vision_config()
        mercedes_config = SensorConfigFactory.create_mercedes_drive_pilot_config()

        # Define functional groups of sensors that serve similar purposes
        sensor_groups = {
            "forward_main": {
                "tesla": ["main_forward_camera", "forward_wide_camera", "forward_narrow_camera"],
                "mercedes": ["stereo_multi_purpose_camera", "long_range_camera", "front_surround_camera"]
            },
            "left_side": {
                "tesla": ["left_b_pillar_camera", "left_front_fender_camera"],
                "mercedes": ["left_surround_camera"]
            },
            "right_side": {
                "tesla": ["right_b_pillar_camera", "right_front_fender_camera"],
                "mercedes": ["right_surround_camera"]
            },
            "rear": {
                "tesla": ["rear_camera"],
                "mercedes": ["rear_surround_camera"]
            },
            "front_radar": {
                "tesla": [],
                "mercedes": ["front_long_range_radar", "front_left_corner_radar", "front_right_corner_radar"]
            },
            "rear_radar": {
                "tesla": [],
                "mercedes": ["rear_left_corner_radar", "rear_right_corner_radar"]
            },
            "lidar": {
                "tesla": [],
                "mercedes": ["roof_lidar"]
            },
            "special": {
                "tesla": [],
                "mercedes": ["night_vision_camera"]
            }
        }

        # Create a new hybrid configuration
        hybrid_config = {}

        # For each sensor group, randomly choose between Tesla and Mercedes
        for group_name, options in sensor_groups.items():
            # If both manufacturers have sensors for this group
            if options["tesla"] and options["mercedes"]:
                # Randomly choose manufacturer
                chosen_manufacturer = random.choice(["tesla", "mercedes"])
                chosen_sensors = options[chosen_manufacturer]

                # Add the chosen sensors to the hybrid config
                for sensor_name in chosen_sensors:
                    if chosen_manufacturer == "tesla":
                        hybrid_config[sensor_name] = tesla_config[sensor_name]
                    else:
                        hybrid_config[sensor_name] = mercedes_config[sensor_name]

            # If only one manufacturer has sensors for this group
            elif options["tesla"]:
                for sensor_name in options["tesla"]:
                    hybrid_config[sensor_name] = tesla_config[sensor_name]
            elif options["mercedes"]:
                # For non-camera sensors, add with some probability
                if group_name in ["front_radar", "rear_radar", "lidar", "special"]:
                    if random.random() < 0.7:  # 70% chance to include these sensor types
                        for sensor_name in options["mercedes"]:
                            hybrid_config[sensor_name] = mercedes_config[sensor_name]
                else:
                    for sensor_name in options["mercedes"]:
                        hybrid_config[sensor_name] = mercedes_config[sensor_name]

        # Count sensor types for logging
        camera_count = sum(1 for name, config in hybrid_config.items() if config["type"] == "camera")
        radar_count = sum(1 for name, config in hybrid_config.items() if config["type"] == "radar")
        lidar_count = sum(1 for name, config in hybrid_config.items() if config["type"] == "lidar")

        print(f"Created hybrid configuration with {camera_count} cameras, {radar_count} radars, {lidar_count} lidars")

        return hybrid_config

    @staticmethod
    def create_sensor_config(config_type="generic_av"):
        """
        Creates a sensor configuration with options for different vehicle types.

        Args:
            config_type (str): Configuration type - "generic_av" (default),
                            "tesla_vision", "mercedes_drive_pilot"

        Returns:
            dict: A dictionary containing sensor configurations
        """
        valid_types = ["generic_av", "tesla_vision", "mercedes_drive_pilot"]
        if config_type not in valid_types:
            raise ValueError(f"Invalid config_type: {config_type}. Must be one of {valid_types}")

        if config_type == "tesla_vision":
            return SensorConfigFactory.create_tesla_vision_config()
        elif config_type == "mercedes_drive_pilot":
            return SensorConfigFactory.create_mercedes_drive_pilot_config()
        else:  # generic_av is the default
            return SensorConfigFactory.create_generic_av_config()


class SimulationUtils:
    """
    Utility functions for simulation.
    """
    @staticmethod
    def convert_sensor_config_to_list(sensor_config):
        """
        Convert dictionary sensor config to list format for simulation.

        Args:
            sensor_config (dict): Dictionary containing sensor configurations

        Returns:
            list: List of sensor configurations in the format expected by simulation functions
        """
        sensor_configs = []
        for name, config in sensor_config.items():
            sensor_dict = {
                'name': name,
                'type': config['type'],
                'position': tuple(config['position']),
                'range': config['range'],
                'fov': config.get('field_of_view', config.get('horizontal_fov', 120)),
                'orientation': tuple(config['orientation'])
            }

            # Add sensor-specific parameters
            if config['type'] == 'lidar' and 'k' in config:
                sensor_dict['k'] = config['k']

            if config['type'] == 'radar':
                if 'a' in config:
                    sensor_dict['a'] = config['a']
                if 'd0' in config:
                    sensor_dict['d0'] = config['d0']

            sensor_configs.append(sensor_dict)
        return sensor_configs

    @staticmethod
    def generate_valid_coordinate(min_val, max_val, exclusion_min, exclusion_max):
        """
        Generate a random coordinate that is outside the exclusion range.

        Args:
            min_val (float): Minimum allowed value
            max_val (float): Maximum allowed value
            exclusion_min (float): Minimum value of exclusion range
            exclusion_max (float): Maximum value of exclusion range

        Returns:
            float: A random coordinate outside the exclusion range
        """
        while True:
            val = random.uniform(min_val, max_val)
            if val < exclusion_min or val > exclusion_max:
                return val

    @staticmethod
    def create_object_detection_simulation(num_objects=50, seed=42):
        """
        Create a simulation with randomly placed objects in a defined area.

        Args:
            num_objects: Number of objects to place in the simulation
            seed: Random seed for reproducibility

        Returns:
            List of objects with their positions and types
        """
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        objects = []
        object_types = ['pedestrian', 'vehicle', 'cyclist', 'static_obstacle']
        object_type_weights = [0.1, 0.6, 0.2, 0.1]  # Probability distribution for object types

        for _ in range(num_objects):
            obj_type = random.choices(object_types, weights=object_type_weights, k=1)[0]

            # Generate valid x coordinate (outside the range [-2.7, 2.7])
            x = SimulationUtils.generate_valid_coordinate(-40, 150, -2.7, 2.7)

            # Generate valid y coordinate (outside the range [-1, 1])
            y = SimulationUtils.generate_valid_coordinate(-40, 40, -1, 1)

            z = 1

            objects.append({
                'type': obj_type,
                'position': (x, y, z),
                'id': str(uuid.uuid4())  # Add unique ID for tracking
            })

        return objects

    @staticmethod
    def calculate_detection_probability(sensor, obj, visibility_factor=1.0, environment_conditions=None):
        """
        Calculates detection probability for a given sensor-object pair, incorporating:
        - FOV and range checks.
        - Sensor-specific probability decay (camera: binary, lidar: exponential, radar: sigmoid).
        - Sensor noise.
        - Sensor dropout.
        - Environmental conditions (fog).

        Args:
            sensor (dict): Sensor configuration
            obj (dict): Object to detect
            visibility_factor (float): Factor affecting visibility (1.0 = normal)
            environment_conditions (dict): Environmental conditions like fog density

        Returns:
            float: Detection probability (0.0 to 1.0)
        """
        if environment_conditions is None:
            environment_conditions = {'fog_density': 0.0}

        # Extract positions
        sx, sy, sz = sensor['position']
        tx, ty, tz = obj['position']

        # --- Basic Geometry Check ---
        distance = np.sqrt((sx - tx)**2 + (sy - ty)**2 + (sz - tz)**2)

        if distance > sensor['range']:
            return 0.0

        # Field of view check
        if 'fov' in sensor and sensor['fov'] < 360:
            # Calculate horizontal angle (azimuth)
            horizontal_angle = math.degrees(math.atan2(ty - sy, tx - sx))
            sensor_yaw = sensor['orientation'][2]
            relative_horizontal = horizontal_angle - sensor_yaw

            # Normalize to -180 to 180
            while relative_horizontal > 180:
                relative_horizontal -= 360
            while relative_horizontal < -180:
                relative_horizontal += 360

            # Check horizontal FOV
            if abs(relative_horizontal) > sensor['fov'] / 2:
                return 0.0

            # Calculate vertical angle (elevation)
            # We need the horizontal distance for this
            horizontal_distance = math.sqrt((tx - sx)**2 + (ty - sy)**2)
            vertical_angle = math.degrees(math.atan2(tz - sz, horizontal_distance))

            # Get sensor pitch (if available, otherwise assume 0)
            sensor_pitch = sensor['orientation'][1] if len(sensor['orientation']) > 1 else 0
            relative_vertical = vertical_angle - sensor_pitch

            # Normalize to -90 to 90 (vertical angles are between -90 and 90 degrees)
            if relative_vertical > 90:
                relative_vertical = 180 - relative_vertical
            elif relative_vertical < -90:
                relative_vertical = -180 - relative_vertical

            # Check vertical FOV (if defined, otherwise use a default or derived value)
            vertical_fov = sensor.get('vertical_fov', sensor['fov'] / 2)  # Default: half of horizontal FOV

            if abs(relative_vertical) > vertical_fov / 2:
                return 0.0

        # --- Base Detection Probability by Sensor Type ---
        sensor_type = sensor['type'].lower()

        # Default value in case parameters are missing
        detection_prob = 1.0

        if sensor_type == 'camera':
            # Cameras: Binary detection if within range/FOV
            detection_prob = 1.0

        elif sensor_type == 'lidar':
            # Lidar: Exponential decay with distance
            k = sensor.get('k', 0.01)  # Default decay rate if not specified
            detection_prob = np.exp(-k * distance)

        elif sensor_type == 'radar':
            # Radar: Sigmoid decay with distance
            a = sensor.get('a', 0.1)     # Steepness parameter
            d0 = sensor.get('d0', sensor['range'] * 0.8)  # Midpoint at 80% of range by default
            detection_prob = 1 / (1 + np.exp(a * (distance - d0)))

        else:
            # Default for unknown sensor types
            detection_prob = 1.0

        # --- Apply Fog Effects ---
        fog_density = environment_conditions.get('fog_density', 0.0)

        if fog_density > 0:
            # Convert fog_density (0-1) to meteorological extinction coefficient (β)
            # Based on Gultepe et al. (2007)
            # Light fog (0.2): visibility ~1000m (β=0.003)
            # Moderate fog (0.5): visibility ~300m (β=0.01)
            # Heavy fog (0.8): visibility ~100m (β=0.03)
            # Very dense fog (1.0): visibility ~50m (β=0.06)

            # Meteorological extinction coefficient (β) calculation
            beta_vis = 0.003 + 0.057 * fog_density**2  # Nonlinear relationship

            # Apply Beer-Lambert law for fog attenuation based on sensor type
            if sensor_type == 'camera':
                # Cameras severely affected by fog (Hasirlioglu & Riener, 2020)
                camera_factor = 1.0  # Full effect
                fog_attenuation = np.exp(-beta_vis * camera_factor * distance)
                detection_prob *= fog_attenuation

            elif sensor_type == 'lidar':
                # Lidar moderately affected by fog (Bijelic et al., 2018)
                # 905nm wavelength (common in automotive)
                lidar_factor = 0.7  # 70% of visual extinction
                fog_attenuation = np.exp(-beta_vis * lidar_factor * distance)
                detection_prob *= fog_attenuation

            elif sensor_type == 'radar':
                # Radar minimally affected by fog (Brooker, 2007)
                # 77GHz automotive radar
                radar_factor = 0.05  # Only 5% of visual extinction
                fog_attenuation = np.exp(-beta_vis * radar_factor * distance)
                detection_prob *= fog_attenuation

        # Clip to [0,1] (just in case numerical errors push it outside)
        detection_prob = max(0.0, min(1.0, detection_prob * visibility_factor))

        # --- Sensor Noise (Gaussian) ---
        # Bar-Shalom et al. 2001: Zero-mean Gaussian noise typical
        if sensor_type == 'camera':
            sigma = 0.03
        elif sensor_type == 'lidar':
            sigma = 0.05
        elif sensor_type == 'radar':
            sigma = 0.02
        else:
            sigma = 0.04  # default

        noise = np.random.normal(0, sigma)
        detection_prob += noise

        detection_prob = max(0.0, min(1.0, detection_prob))

        # --- Sensor Dropout (Bernoulli) ---
        dropout_rates = {
            'camera': 0.001,
            'lidar': 0.005,
            'radar': 0.0005
        }

        dropout_chance = dropout_rates.get(sensor_type, 0.002)

        if random.random() < dropout_chance:
            detection_prob = 0.0

        return detection_prob

    @staticmethod
    def sensor_fusion_detection(sensors, obj, all_objects, environment_conditions=None):
        """
        Perform sensor fusion to detect an object.

        Args:
            sensors (list): List of sensor configurations
            obj (dict): Object to detect
            all_objects (list): All objects in the simulation
            environment_conditions (dict): Environmental conditions like fog density

        Returns:
            tuple: (detected, detecting_sensors, detections_by_type)
                detected (bool): Whether the object was detected
                detecting_sensors (dict): Dictionary of sensors that detected the object
                detections_by_type (dict): Count of detections by sensor type
        """
        if environment_conditions is None:
            environment_conditions = {'fog_density': 0.0}

        # Track detections by sensor type
        detections_by_type = {}
        detected = False
        detecting_sensors = {}

        # Step 1: Check each sensor
        for sensor in sensors:
            # Calculate detection
            detection_prob = SimulationUtils.calculate_detection_probability(
                sensor, obj, 1.0, environment_conditions
            )

            # If sensor detects the object
            detection_threshold = 0.0  # Adjust as needed
            if detection_prob > detection_threshold:
                detected = True
                sensor_type = sensor['type']
                detections_by_type[sensor_type] = detections_by_type.get(sensor_type, 0) + 1
                detecting_sensors[sensor['name']] = detection_prob  # Store the probability

        return detected, detecting_sensors, detections_by_type


class SimulationRunner:
    """
    Class for running vehicle sensor simulations.
    """
    @staticmethod
    def run_monte_carlo_simulation_with_fusion(sensor_configs, num_objects=50,
                                            num_iterations=100,
                                            seed=42, environment_conditions=None):
        """
        Run a fixed number of Monte Carlo simulation iterations using sensor fusion for detection decisions.

        Args:
            sensor_configs (list): List of sensor configurations
            num_objects (int): Number of objects to place in the simulation
            num_iterations (int): Number of simulation iterations to run
            seed (int): Random seed for reproducibility
            environment_conditions (dict): Environmental conditions like fog density

        Returns:
            dict: Simulation results
        """
        if environment_conditions is None:
            environment_conditions = {'fog_density': 0.0}

        detection_rates = []
        detection_by_type = {}
        object_type_totals_per_iter = {}
        all_iteration_data = []

        # Track sensor type detection percentages for each iteration
        sensor_type_detection_percentages = {
            'camera': [],
            'lidar': [],
            'radar': []
        }

        # Count sensor types in configuration
        sensor_type_counts = {
            'camera': sum(1 for s in sensor_configs if s['type'].lower() == 'camera'),
            'lidar': sum(1 for s in sensor_configs if s['type'].lower() == 'lidar'),
            'radar': sum(1 for s in sensor_configs if s['type'].lower() == 'radar')
        }

        # Run for fixed number of iterations
        for i in range(num_iterations):
            iter_seed = seed + i

            # Create simulation objects for this iteration
            objects = SimulationUtils.create_object_detection_simulation(num_objects, seed=iter_seed)
            last_iteration_objects = objects if i == num_iterations - 1 else None

            # Initialize counts for object types for THIS iteration
            current_iter_type_counts = {}
            for obj in objects:
                obj_type = obj['type']
                current_iter_type_counts[obj_type] = current_iter_type_counts.get(obj_type, 0) + 1
                if obj_type not in object_type_totals_per_iter:
                    object_type_totals_per_iter[obj_type] = 0
                object_type_totals_per_iter[obj_type] += 1

            # Run detection with sensor fusion for this iteration
            detected_this_iter = set()
            detected_by_type_this_iter = {}
            iter_detections = []

            # Track which objects are detected by each sensor type in this iteration
            objects_detected_by_type = {
                'camera': set(),
                'lidar': set(),
                'radar': set()
            }

            # Process each object with all sensors at once (fusion approach)
            for obj in objects:
                # Create a list of objects excluding the current target to prevent self-occlusion
                other_objects = [o for o in objects if o['id'] != obj['id']]

                is_detected, sensor_probs, detecting_sensor_types = SimulationUtils.sensor_fusion_detection(
                    sensor_configs, obj, other_objects, environment_conditions
                )

                # Update which objects are detected by each sensor type
                if is_detected:
                    for s_type in detecting_sensor_types:
                        if s_type in objects_detected_by_type:
                            objects_detected_by_type[s_type].add(obj['id'])

                # Store detection result
                iter_detections.append({
                    'object_id': obj['id'],
                    'object_type': obj['type'],
                    'detected': is_detected,
                    'sensor_probabilities': sensor_probs,
                    'detecting_sensor_types': detecting_sensor_types if is_detected else []
                })

                if is_detected:
                    detected_this_iter.add(obj['id'])
                    obj_type = obj['type']
                    detected_by_type_this_iter[obj_type] = detected_by_type_this_iter.get(obj_type, 0) + 1

            # Calculate detection rate for this iteration
            detection_rate = len(detected_this_iter) / num_objects if num_objects > 0 else 0
            detection_rates.append(detection_rate)

            # Calculate and store detection percentages by sensor type for this iteration
            for s_type in objects_detected_by_type:
                if sensor_type_counts[s_type] > 0:  # Only calculate if this sensor type exists
                    percentage = len(objects_detected_by_type[s_type]) / num_objects if num_objects > 0 else 0
                    sensor_type_detection_percentages[s_type].append(percentage)

            # Update cumulative detection counts by type
            for obj_type, count in detected_by_type_this_iter.items():
                if obj_type not in detection_by_type:
                    detection_by_type[obj_type] = {'total': 0, 'detected': 0}
                detection_by_type[obj_type]['detected'] += count

            # Store data for this iteration
            all_iteration_data.append({
                'iteration': i,
                'detection_rate': detection_rate,
                'detections_by_type': detected_by_type_this_iter.copy(),
                'total_by_type': current_iter_type_counts.copy(),
                'detailed_detections': iter_detections,
                'objects_detected_by_type': {k: len(v) for k, v in objects_detected_by_type.items()}
            })

        # Calculate final statistics
        mean_detection_rate = np.mean(detection_rates)

        # Finalize detection rate by object type
        final_detection_by_type = {}
        for obj_type, counts in detection_by_type.items():
            total_instances_of_type = object_type_totals_per_iter.get(obj_type, 0)
            detected_count = counts['detected']
            rate = detected_count / total_instances_of_type if total_instances_of_type > 0 else 0.0
            final_detection_by_type[obj_type] = {
                'total': total_instances_of_type,
                'detected': detected_count,
                'rate': rate
            }

        # Calculate mean detection percentages by sensor type
        mean_sensor_type_percentages = {}
        for s_type, percentages in sensor_type_detection_percentages.items():
            if sensor_type_counts[s_type] > 0 and percentages:  # Only include if this sensor type exists and has data
                mean_sensor_type_percentages[s_type] = np.mean(percentages)
            else:
                mean_sensor_type_percentages[s_type] = 0.0

        return {
            'mean_detection_rate': mean_detection_rate,
            'iterations_run': num_iterations,
            'individual_rates': detection_rates,
            'detection_by_type': final_detection_by_type,
            'sensors': sensor_configs,
            'objects': last_iteration_objects,
            'total_objects': num_objects,
            'objects_detected': int(mean_detection_rate * num_objects),
            'detection_rate': mean_detection_rate,
            'all_iteration_data': all_iteration_data,
            'mean_sensor_type_percentages': mean_sensor_type_percentages,
            'sensor_type_counts': sensor_type_counts
        }


class VisualizationUtils:
    """
    Utility functions for visualizing simulation results.
    """
    @staticmethod
    def visualize_fog_effects(config_types=["tesla_vision", "mercedes_drive_pilot", "generic_av"],
                           fog_levels=[0.0, 0.3, 0.6, 0.9], num_iterations=1000):
        """
        Visualize the effect of fog on different sensor configurations.

        Args:
            config_types (list): List of sensor configuration types to test
            fog_levels (list): List of fog density levels to test
            num_iterations (int): Number of simulation iterations to run

        Returns:
            dict: Results of fog effect simulations
        """
        results = {}

        for config_type in config_types:
            fog_results = []
            sensor_config = SensorConfigFactory.create_sensor_config(config_type)
            sensor_configs = SimulationUtils.convert_sensor_config_to_list(sensor_config)

            # Run simulation with different fog levels
            for fog_density in fog_levels:
                print(f"Running {config_type} with fog density {fog_density}")
                result = SimulationRunner.run_monte_carlo_simulation_with_fusion(
                    sensor_configs=sensor_configs,
                    num_objects=20,
                    num_iterations=num_iterations,
                    environment_conditions={'fog_density': fog_density}
                )

                # Calculate sensor type percentages
                sensor_type_percentages = {}
                sensor_type_detections = {}

                # Extract sensor type detection counts from results
                for iteration_data in result['all_iteration_data']:
                    for s_type, count in iteration_data.get('objects_detected_by_type', {}).items():
                        sensor_type_detections[s_type] = sensor_type_detections.get(s_type, 0) + count

                total_detections = sum(sensor_type_detections.values())

                if total_detections > 0:
                    for s_type in sensor_type_detections:
                        sensor_type_percentages[s_type] = sensor_type_detections[s_type] / total_detections

                fog_results.append({
                    'fog_density': fog_density,
                    'detection_rate': result['mean_detection_rate'],
                    'sensor_type_detections': sensor_type_detections,
                    'sensor_type_percentages': sensor_type_percentages
                })

            results[config_type] = fog_results

        # Plot overall detection rates
        plt.figure(figsize=(12, 6))

        for config_type, fog_results in results.items():
            fog_densities = [r['fog_density'] for r in fog_results]
            detection_rates = [r['detection_rate'] for r in fog_results]

            plt.plot(fog_densities, detection_rates, 'o-', linewidth=2,
                    label=f"{config_type}")

        plt.title('Effect of Fog Density on Detection Rate', fontsize=15)
        plt.xlabel('Fog Density', fontsize=12)
        plt.ylabel('Detection Rate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Configuration')
        plt.savefig("fog_effect_overall.png", dpi=300, bbox_inches='tight')
        plt.show()

        return results

    @staticmethod
    def simulate_hybrid_configurations(num_simulations=10, num_iterations=1000, fog_levels=[0.0, 0.4, 0.8]):
        """
        Generate and test multiple hybrid configurations.

        Args:
            num_simulations: Number of different hybrid configurations to test
            num_iterations: Number of iterations per configuration
            fog_levels: Fog density levels to test

        Returns:
            tuple: (results, best_config, hybrid_config)
                results (dict): Results for all simulated configurations
                best_config (str): Name of the best performing configuration
                hybrid_config (dict): Configuration of the best performing hybrid
        """
        results = {}
        best_config = None
        best_performance = 0
        best_hybrid_config = None

        for sim_num in range(num_simulations):
            print(f"\n=== Testing Hybrid Configuration #{sim_num+1} ===")

            # Create a hybrid configuration
            hybrid_config = SensorConfigFactory.create_hybrid_configuration()
            config_name = f"hybrid_{sim_num+1}"

            # Convert configuration to list format for simulation
            sensor_configs = SimulationUtils.convert_sensor_config_to_list(hybrid_config)

            # Test this configuration across fog levels
            fog_results = []
            avg_performance = 0

            for fog_density in fog_levels:
                print(f"  Testing with fog density {fog_density:.1f}...")

                result = SimulationRunner.run_monte_carlo_simulation_with_fusion(
                    sensor_configs=sensor_configs,
                    num_objects=20,
                    num_iterations=num_iterations,
                    environment_conditions={'fog_density': fog_density}
                )

                detection_rate = result['mean_detection_rate']
                print(f"  Fog density {fog_density:.1f}: Detection rate {detection_rate:.4f}")

                fog_results.append({
                    'fog_density': fog_density,
                    'detection_rate': detection_rate
                })

                avg_performance += detection_rate

            # Calculate average performance across all fog levels
            avg_performance /= len(fog_levels)
            print(f"  Average detection rate across all fog levels: {avg_performance:.4f}")

            # Store results
            results[config_name] = {
                'config': hybrid_config,
                'fog_results': fog_results,
                'avg_performance': avg_performance
            }

            # Track best configuration
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_config = config_name
                best_hybrid_config = hybrid_config

        # Compare with baseline configurations
        print("\n=== Comparing with Baseline Configurations ===")

        for config_type in ["tesla_vision", "mercedes_drive_pilot"]:
            print(f"\nTesting {config_type}...")

            sensor_config = SensorConfigFactory.create_sensor_config(config_type)
            sensor_configs = SimulationUtils.convert_sensor_config_to_list(sensor_config)

            fog_results = []
            avg_performance = 0

            for fog_density in fog_levels:
                result = SimulationRunner.run_monte_carlo_simulation_with_fusion(
                    sensor_configs=sensor_configs,
                    num_objects=20,
                    num_iterations=num_iterations,
                    environment_conditions={'fog_density': fog_density}
                )

                detection_rate = result['mean_detection_rate']
                print(f"  Fog density {fog_density:.1f}: Detection rate {detection_rate:.4f}")

                fog_results.append({
                    'fog_density': fog_density,
                    'detection_rate': detection_rate
                })

                avg_performance += detection_rate

            avg_performance /= len(fog_levels)
            print(f"  Average detection rate across all fog levels: {avg_performance:.4f}")

            results[config_type] = {
                'fog_results': fog_results,
                'avg_performance': avg_performance
            }

        # Print final comparison
        print("\n=== Final Results ===")
        print(f"Best hybrid configuration: {best_config}")
        print(f"Best hybrid performance: {results[best_config]['avg_performance']:.4f}")
        print(f"Tesla Vision performance: {results['tesla_vision']['avg_performance']:.4f}")
        print(f"Mercedes Drive Pilot performance: {results['mercedes_drive_pilot']['avg_performance']:.4f}")

        # Plot performance comparison
        plt.figure(figsize=(12, 6))

        # Plot hybrid configurations
        for config_name, result in results.items():
            if config_name.startswith('hybrid'):
                fog_densities = [r['fog_density'] for r in result['fog_results']]
                detection_rates = [r['detection_rate'] for r in result['fog_results']]

                if config_name == best_config:
                    plt.plot(fog_densities, detection_rates, 'o-', linewidth=3,
                            label=f"{config_name} (BEST)")
                else:
                    plt.plot(fog_densities, detection_rates, '--', alpha=0.3,
                            label=config_name)

        # Plot baseline configurations
        for config_type in ["tesla_vision", "mercedes_drive_pilot"]:
            result = results[config_type]
            fog_densities = [r['fog_density'] for r in result['fog_results']]
            detection_rates = [r['detection_rate'] for r in result['fog_results']]

            plt.plot(fog_densities, detection_rates, 'o-', linewidth=2,
                    label=config_type)

        plt.title('Comparison of Sensor Configurations', fontsize=15)
        plt.xlabel('Fog Density', fontsize=12)
        plt.ylabel('Detection Rate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Configuration')
        plt.savefig("hybrid_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

        return results, best_config, best_hybrid_config

    @staticmethod
    def plot_convergence(results_by_config):
        """
        Plot the convergence of detection rates over iterations.

        Args:
            results_by_config (dict): Dictionary mapping configuration names to simulation results
        """
        plt.figure(figsize=(10, 6))

        for config, results in results_by_config.items():
            rates = results['individual_rates']
            iterations = range(1, len(rates) + 1)

            # Calculate cumulative moving average to visualize convergence
            cumulative_avg = np.cumsum(rates) / np.arange(1, len(rates) + 1)

            plt.plot(iterations, cumulative_avg, label=f"{config} (final: {results['mean_detection_rate']:.2%})")

        plt.title('Convergence of Detection Rates Over Iterations', fontsize=15)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cumulative Average Detection Rate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Configuration')
        plt.savefig("convergence_plot.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to run simulations and visualize results.
    """
    config_types = ["tesla_vision", "mercedes_drive_pilot", "generic_av"]
    results_by_config = {}

    for config_type in config_types:
        print(f"\n=== MC for config: {config_type} ===")
        sensor_config = SensorConfigFactory.create_sensor_config(config_type)
        sensor_configs = SimulationUtils.convert_sensor_config_to_list(sensor_config)

        results = SimulationRunner.run_monte_carlo_simulation_with_fusion(
            sensor_configs=sensor_configs,
            num_objects=20,
            num_iterations=5000,  # Fixed number of iterations
            environment_conditions={'fog_density': 0.0}
        )

        results_by_config[config_type] = results
        print(f"- Mean Detection Rate: {results['mean_detection_rate']:.2%}")
        print(f"- Completed {results['iterations_run']} iterations")

    # Plot the convergence using the individual_rates data
    VisualizationUtils.plot_convergence(results_by_config)

    # In the main function, after running all simulations:
    print("\n=== Sensor Type Detection Analysis ===")
    for config_type, results in results_by_config.items():
        print(f"\n{config_type} Configuration:")
        print(f"  Sensor counts: {results['sensor_type_counts']}")
        print("  Average percentage of objects detected by each sensor type:")
        for s_type, percentage in results['mean_sensor_type_percentages'].items():
            if results['sensor_type_counts'][s_type] > 0:  # Only show types that exist in this config
                print(f"    - {s_type}: {percentage:.2%}")

    print("\n=== Analyzing Fog Effects ===")
    fog_results = VisualizationUtils.visualize_fog_effects(
        config_types=["tesla_vision", "mercedes_drive_pilot"],
        fog_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
        num_iterations=1000
    )

    # Print some key findings
    print("\n=== Key Findings on Fog Effects ===")
    for config_type, results in fog_results.items():
        print(f"\n{config_type} Configuration:")
        for r in results:
            print(f"  Fog density {r['fog_density']:.1f}: Detection rate {r['detection_rate']:.2%}")

        # Calculate degradation from clear to dense fog
        clear_rate = results[0]['detection_rate']
        dense_rate = results[-1]['detection_rate']
        degradation = (clear_rate - dense_rate) / clear_rate * 100

        print(f"  Overall degradation: {degradation:.1f}% from clear to dense fog")

    hybrid_results, best_config, best_hybrid = VisualizationUtils.simulate_hybrid_configurations(
        num_simulations=5,  # Test 5 different hybrid configurations
        num_iterations=1000,  # 1000 iterations per configuration
        fog_levels=[0.0, 0.4, 0.8]  # Test with no fog, medium fog, heavy fog
    )


if __name__ == "__main__":
    main()