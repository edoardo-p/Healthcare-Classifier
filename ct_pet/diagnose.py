import numpy as np
import SimpleITK as sitk
import torch
from radiomics import featureextractor
from unet import UNet

radiomics.setVerbosity(40)


def threshold(data: torch.Tensor, level: float = 0.5) -> torch.Tensor:
    scaled = (data - data.min()) / (data.max() - data.min())
    scaled[scaled < level] = 0
    scaled[scaled >= level] = 1
    return scaled


def segment(image: np.ndarray) -> np.ndarray:
    model = UNet(residual=False, cat=True)
    model.load_state_dict(torch.load(".\\models\\conv_l1_cat_500.pt"))

    pred = threshold(model(torch.tensor(image)))
    return pred.cpu().detach().numpy()


def get_haralick_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = sitk.ReadImage(image)
    msk = sitk.ReadImage(mask)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeaturesClassByName("glcm")
    result = extractor.execute(img, msk)
    return result
