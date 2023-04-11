import streamlit as st
from streamlit_image_comparison import image_comparison

from PIL import Image
import albumentations as A

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np

from support import restore_mask, mask_to_img, weights, class_dict

st.set_page_config(
    "Image Segmentation",
    layout="wide"
)

st.markdown("""
<style>
    #the-title {text-align: center}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
</style>
""", unsafe_allow_html=True)

st.write("<h1 style='text-align: center;'>Semantic Segmentation</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>by Antonio Nardi</p>", unsafe_allow_html=True)


## Image Upload
st.header('Image Upload')
base_img_bytes = None

if base_img_bytes is None:
    base_img_bytes = st.file_uploader(
        """
        Image must be taken at street level with the sky at the top 
        of the image and the ground at the bottom. HAVE FUN!
        """, 
        type=['png', 'jpg'], 
        accept_multiple_files=False, 
        key='base_img', 
        help="""
        Please upload the image that will be segmented. After the image
        is processed, it will be deleted from our server.
        """, 
        label_visibility='visible'
    )

## Inference
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

custom_objects = {
    'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss,
    'iou_score': sm.metrics.iou_score,
    'dice_loss_plus_1focal_loss': total_loss,
    'f1-score': sm.metrics.FScore(threshold=0.5)
}

with keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model('unet_model.h5')

st.divider()

if base_img_bytes is not None:

    padding1, content, padding2 = st.columns([35, 30, 35])
    with content:
        st.header('Segmentation Mask')
        progress_bar = st.progress(0, text='Initializing Process')

        base_img = np.asarray(Image.open(base_img_bytes).resize((512,512)))
        progress_bar.progress(10, text='Loaded Image')
        mods = A.Compose([
            # A.CLAHE (clip_limit=3.0, tile_grid_size=(4, 4), always_apply=True)
        ])
        img_input = mods(image=base_img)['image']
        progress_bar.progress(30, text='Processed Image')

        sm_mask_img = model.predict(np.expand_dims(img_input/255, axis=0)).squeeze()
        progress_bar.progress(75, text='Generating Mask')

        mod_m = np.transpose(sm_mask_img, (2, 0, 1))
        unary = unary_from_softmax(mod_m)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2D(512, 512, 66)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=base_img.copy(), compat=10)

        Q = d.inference(40)
        progress_bar.progress(90, text='Processing Mask')

        # Get the refined segmentation mask
        refined_mask = np.argmax(Q, axis=0).reshape((512, 512))
        
        mask_img = mask_to_img(refined_mask, class_dict)
        progress_bar.progress(100, text='Finished Mask')
        progress_bar.empty()

        image_comparison(
            img1=Image.fromarray(base_img),
            img2=mask_img,
            width=700,
            label1 = "Image",
	        label2 = "Mask"
        )
    st.balloons()