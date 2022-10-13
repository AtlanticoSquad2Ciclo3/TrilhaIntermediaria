
import kornia.augmentation as K

aug = K.AugmentationSequential(
   K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
   K.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30., 50.], p=1.0),
   K.RandomPerspective(0.5, p=1.0),
   data_keys=["input", "bbox", "keypoints", "mask"],  # Just to define the future input here.
   return_transform=False,
   same_on_batch=False,
)
# forward the operation
out_tensors = aug(img_tensor, bbox, keypoints, mask)
# Inverse the operation
out_tensor_inv = aug.inverse(*out_tensor)