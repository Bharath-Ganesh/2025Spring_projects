from typing import Dict, Any

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
            "horizontal_fov": 145 ,
            "vertical_fov" : 25,
            "range": 200,
            "resolution": [0.1, 0.1]  # angular resolution in degrees
        },

    }

    return config


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


def create_sensor_config(config_type="generic_av"):
    """
    Creates a sensor configuration with options for different vehicle types.

    Args:
        config_type (str): Configuration type - "generic_av" (default),
                          "tesla_vision", "mercedes_drive_pilot"

    Returns:
        dict: A dictionary containing sensor configurations
    """
    if config_type == "tesla_vision":
        return create_tesla_vision_config()
    elif config_type == "mercedes_drive_pilot":
        return create_mercedes_drive_pilot_config()
    else:  # generic_av is the default
        return create_generic_av_config()
