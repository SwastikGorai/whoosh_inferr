import pandas as pd
import io
import numpy as np
from PIL import Image 


from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors


# model = YOLO('''TODO: path to model''')


###################################################################
####### Image to byte #############################################
###################################################################

def image_to_byte(img) -> bytes:
    """Convert image to byte array"""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=100)
    img_byte_arr.seek(0)
    return img_byte_arr
    


###################################################################
####### Byte to Image #############################################
###################################################################

def byte_to_image(byte) -> Image:
    """Convert byte array to image"""
    img = Image.open(io.BytesIO(byte)).convert('RGB')
    return img


###################################################################
####### Convert Predictions to DataFrame ##########################
###################################################################

def convert_predictions_to_df(predictions: list, labels: dict = {0: "Plastic"}) -> pd.DataFrame:
    """Convert predictions to DataFrame"""
    # df = pd.DataFrame()
    data = predictions[0].to("cpu").numpy().boxes
    df = pd.DataFrame(predictions[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    df["Confidence"] = data.conf
    print((data.cls).astype(int))
    # df["Labels"] = labels[(data.cls).astype(int)] # Maybe ?
    return df


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
    predict= convert_predictions_to_df(predict)
    return predict



###################################################################
#### Add boxes to predictions #####################################
###################################################################

def add_boxes_to_predictions(predictions: pd.DataFrame() , img: Image, labels: dict = {0:"Plastic"}) -> Image:
    """Add boxes to predictions"""
    predictions = predictions.sort_values(by=["xmin"], ascending=True)
    annotator = Annotator(np.array(img))
    for i, row in predictions.iterrows():
        annotator.box_label([row["xmin"], row["ymin"], row["xmax"], row["ymax"]], 
                            "Plastic")
    res = annotator.result()
    return Image.fromarray(res)

