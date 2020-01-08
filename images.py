from io import BytesIO
from os import path
import requests
from PIL import Image as _Image
from torchvision.transforms.functional import center_crop

__all__ = [
    "get_npr_general_files",
    "get_npr_general_proxy_file",
    "get_style_image_files",
]


CROP_SIZE = (640, 1024)


class Image:
    def __init__(self, title, author, url, center_crop=True):
        self.title = title
        self.author = author
        self.url = url
        self.center_crop = center_crop

    @property
    def file(self):
        name = self.title.lower().replace(" ", "_")
        return name + ".jpg"

    def download(self, root):
        print(f"Downloading image '{self.title}' by '{self.author}' from {self.url}")
        content = BytesIO(requests.get(self.url).content)
        image = _Image.open(content)

        if self.center_crop:
            print(f"Center-cropping image to {CROP_SIZE[1]} x {CROP_SIZE[0]} pixels")
            image = center_crop(image, CROP_SIZE)

        print(f"Saving image as {self.file}")
        image.save(path.join(root, self.file))


npr_general_dataset = [
    Image(
        title="angel",
        author="Eole Wind",
        url="http://gigl.scs.carleton.ca/sites/default/files/angel1024.jpg",
    ),
    Image(
        title="arch",
        author="James Marvin Phelps",
        url="http://gigl.scs.carleton.ca/sites/default/files/arch1024.jpg",
    ),
    Image(
        title="athletes",
        author="Nathan Congleton",
        url="http://gigl.scs.carleton.ca/sites/default/files/athletes1024.jpg",
    ),
    Image(
        title="barn",
        author="MrClean1982",
        url="http://gigl.scs.carleton.ca/sites/default/files/barn1024.jpg",
    ),
    Image(
        title="berries",
        author="HelmutZen",
        url="http://gigl.scs.carleton.ca/sites/default/files/berries1024.jpg",
    ),
    Image(
        title="cabbage",
        author="Leonard Chien",
        url="http://gigl.scs.carleton.ca/sites/default/files/cabbage1024.jpg",
    ),
    Image(
        title="cat",
        author="Theen Moy",
        url="http://gigl.scs.carleton.ca/sites/default/files/cat1024.jpg",
    ),
    Image(
        title="city",
        author="Rob Schneider",
        url="http://gigl.scs.carleton.ca/sites/default/files/city1024.jpg",
    ),
    Image(
        title="daisy",
        author="mgaloseau",
        url="http://gigl.scs.carleton.ca/sites/default/files/daisy1024.jpg",
    ),
    Image(
        title="dark woods",
        author="JB Banks",
        url="http://gigl.scs.carleton.ca/sites/default/files/darkwoods1024.jpg",
    ),
    Image(
        title="desert",
        author="Charles Roffey",
        url="http://gigl.scs.carleton.ca/sites/default/files/desert1024.jpg",
    ),
    Image(
        title="headlight",
        author="Photos By Clark",
        url="http://gigl.scs.carleton.ca/sites/default/files/headlight1024.jpg",
    ),
    Image(
        title="mac",
        author="Martin Kenney",
        url="http://gigl.scs.carleton.ca/sites/default/files/mac1024.jpg",
    ),
    Image(
        title="mountains",
        author="Jenny Pansing (jjjj56cp)",
        url="http://gigl.scs.carleton.ca/sites/default/files/mountains1024.jpg",
    ),
    Image(
        title="oparara",
        author="trevorklatko",
        url="http://gigl.scs.carleton.ca/sites/default/files/oparara1024.jpg",
    ),
    Image(
        title="rim lighting",
        author="Paul Stevenson",
        url="http://gigl.scs.carleton.ca/sites/default/files/rim1024.jpg",
    ),
    Image(
        title="snow",
        author="John Anes",
        url="http://gigl.scs.carleton.ca/sites/default/files/snow1024.jpg",
    ),
    Image(
        title="tomatoes",
        author="Greg Myers",
        url="http://gigl.scs.carleton.ca/sites/default/files/tomato1024.jpg",
    ),
    Image(
        title="toque",
        author="sicknotepix",
        url="http://gigl.scs.carleton.ca/sites/default/files/toque1024.jpg",
    ),
    Image(
        title="yemeni",
        author="Richard Messenger",
        url="http://gigl.scs.carleton.ca/sites/default/files/yemeni1024.jpg",
    ),
]

style_images = [
    Image(
        title="White Zig Zags",
        author="Wassily Kandinsky",
        url="http://www.wassily-kandinsky.org/images/gallery/White-Zig-Zags.jpg",
        center_crop=False,
    ),
    Image(
        title="Landscape at Saint-Remy",
        author="Vincent van Gogh",
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Gogh%2C_Vincent_van_-_Landscape_at_Saint-R%C3%A9my_%28Enclosed_Field_with_Peasant%29_-_Google_Art_Project.jpg/1276px-Gogh%2C_Vincent_van_-_Landscape_at_Saint-R%C3%A9my_%28Enclosed_Field_with_Peasant%29_-_Google_Art_Project.jpg",
        center_crop=False,
    ),
]

images = npr_general_dataset + style_images


def get_npr_general_files():
    return [image.file for image in npr_general_dataset]


def get_npr_general_proxy_file():
    for image in npr_general_dataset:
        if image.title == "berries":
            return image.file


def get_style_image_files():
    return [image.file for image in style_images]


if __name__ == "__main__":
    root = path.dirname(__file__)
    images_root = path.join(root, "images")

    for image in images:
        image.download(images_root)
        print("-" * 200)
