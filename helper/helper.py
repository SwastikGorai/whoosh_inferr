import pandas as pd
import io
import numpy as np
from PIL import Image 


from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors


model = YOLO('''TODO: path to model''')


###################################################################
####### Image to byte #############################################
###################################################################

def image_to_byte(img) -> bytes:
    """Convert image to byte array"""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=100)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


###################################################################
####### Byte to Image #############################################
###################################################################

def byte_to_image(byte) -> Image:
    """Convert byte array to image"""
    img = Image.open(io.BytesIO(byte)).convert('RGB')
    return img


###################################################################
####### Get Predictions ###########################################
###################################################################

def get_predictions(img: Image, model : YOLO, save: bool = False, imgsize : int = 640, conf: float= 0.45, ) -> pd.DataFrame:
    """Get predictions from image"""
    
    predict = model.predict(source=img,
                            imgsz=imgsize,
                            conf=conf,
                            save=save
                        )
    return predict


###################################################################
####### Convert Predictions to DataFrame ##########################
###################################################################

def convert_predictions_to_df(predictions: list, labels: dict = {0: "Plastic"}) -> pd.DataFrame:
    """Convert predictions to DataFrame"""
    df = pd.DataFrame()
    data = predictions[0].to("cpu").numpy().boxes
    df["xmin", "ymin", "xmax", "ymax"] = data.xyxy
    df["Confidence"] = data.conf
    df["Labels"] = labels[(data.cls).astype(int)] # Maybe ?
    return df

###################################################################
#### Add boxes to predictions #####################################
###################################################################

def add_boxes_to_predictions(predictions: list, labels: dict, img: Image) -> pd.DataFrame:
    """Add boxes to predictions"""
    predictions = predictions.sort_values(by="xmin", ascending=False)
    annotator = Annotator(np.array())
    for i, row in predictions.iterrows():
        annotator.box_label([row["xmin"], row["ymin"], row["xmax"], row["ymax"]], 
                            labels[row["Labels"]], 
                            color=colors(row["Labels"]))
    res = annotator.result()
    return Image.fromarray(res)

