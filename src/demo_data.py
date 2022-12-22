import os
import supervisely as sly

demo_weights_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"


def prepare_weights():
    if sly.is_production():
        pass
    else:
        weights_path = os.path.join(sly.env.folder(), "weights.pkl")
        if sly.fs.file_exists(weights_path):
            print(f"Demo NN weights: {weights_path}")
        else:
            print("Downloading model weights, please wait a bit ...")
            sly.fs.download(demo_weights_url, weights_path)
            print(f"Demo NN weights downloaded: {weights_path}")
