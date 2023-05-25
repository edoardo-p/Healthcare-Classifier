from PIL import Image


class BoundingBox:
    def __init__(self, size: int):
        self.left = 0
        self.top = 0
        self.width = self.height = size

    def get_box(self, img: Image.Image, aspect_ratio) -> dict[str, int]:
        print("Called get_box")
        print(self.left + self.width, self.top + self.height)
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }
