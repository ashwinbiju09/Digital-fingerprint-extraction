from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime, timedelta


def time_check(date_taken_str):
    if date_taken_str is not None:
        # Convert the date string to a datetime object
        date_taken = datetime.strptime(date_taken_str, "%Y:%m:%d %H:%M:%S")

        # Get the current datetime
        current_datetime = datetime.now()

        # Calculate the time difference
        time_difference = current_datetime - date_taken

        # Check if the time difference is within 12 hours
        if 0 <= time_difference.total_seconds() <= 12 * 60 * 60:
            return True

    return False
