class FOD:
    def __init__(self, frame_id, bbox, type):
        self.frame_id = frame_id
        self.bbox = bbox
        self.type = type
        self.rotation = 0
        self.scaling = 1

    def scale(self, scale_factor):
        self.scaling *= scale_factor

    def rotate(self, rotate_degree):
        self.rotation += rotate_degree
