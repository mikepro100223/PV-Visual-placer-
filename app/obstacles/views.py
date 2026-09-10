"""Abbas full-image + bounded overlapping crops, at Yucan's obstacle scale."""
import math
from PIL import Image


def detection_views(image,tile=512):
    if max(image.size)>768:
        yield image,0,0
    tile=max(tile,math.ceil(max(image.size)/3))
    stride=tile-tile//4
    def starts(length):
        return sorted(set([*range(0,max(length-tile,0)+1,stride),max(length-tile,0)]))
    for y in starts(image.height):
        for x in starts(image.width):
            crop=image.crop((x,y,min(image.width,x+tile),min(image.height,y+tile)))
            # Retain Yucan's training scale on houses smaller than one crop.
            padded=Image.new('RGB',(tile,tile),(114,114,114))
            padded.paste(crop,(0,0))
            yield padded,x,y
