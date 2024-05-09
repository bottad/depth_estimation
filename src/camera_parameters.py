import json

#######################################################################################
###                              main                                               ###
#######################################################################################

def main():

    sensor_width_mm = 12.8          # Sensor width in mm
    sensor_height_mm = 11.6         # Sensor height in mm
    image_width_pixels = 2000       # Image width in pixels
    image_height_pixels = 1121      # Image height in pixels
    focal_length_mm = 2.97          # Focal length in mm

    baseline = 130                  # Stereo baseline in mm

    # Calculate pixel size
    pixel_size_horizontal = sensor_width_mm / image_width_pixels
    pixel_size_vertical = sensor_height_mm / image_height_pixels

    # Calculate focal length in pixels
    focal_length_pixels_horizontal = focal_length_mm / pixel_size_horizontal
    focal_length_pixels_vertical = focal_length_mm / pixel_size_vertical

    # Take average of horizontal and vertical focal lengths
    focal_length_pixels = (focal_length_pixels_horizontal + focal_length_pixels_vertical) / 2

    print("Focal length in pixels:", focal_length_pixels)

    # Create a dictionary
    camera_params = {
        "FOCAL_LENGTH": focal_length_pixels,
        "BASELINE": baseline/1000
    }

    # Write the dictionary to a JSON file
    filename = "data/camera_parameters.json"
    with open(filename, "w") as file:
        json.dump(camera_params, file)

    print("Camera parameters written to", filename)



if __name__ == '__main__': 
    main()