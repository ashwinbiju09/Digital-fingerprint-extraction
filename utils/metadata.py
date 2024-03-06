from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime, timedelta


def get_metadata(image_path):
    # Open the image
    image = Image.open(image_path)

    # Extract the EXIF data
    exifdata = image.getexif()

    # Look for the DateTimeOriginal tag
    date_taken = None
    for tagid, value in exifdata.items():
        tagname = TAGS.get(tagid, tagid)
        if tagname == "DateTime":
            date_taken = value
            print(date_taken)
            break

    # Close the image
    image.close()

    return date_taken
