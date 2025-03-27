from tqdm import tqdm
import cv2
from tqdm import tqdm

import kornia
import torch


GREEN = (0,255,0)
IMAGE = 'data/kn_church-2.jpg'

def main():
  
  img = cv2.imread(IMAGE)


  with open('data/kn_church-2.sift.1', 'r') as f:
    lns = f.readlines()
  
  feature_num, feature_size = [int(_) for _ in lns[0].split()]
  for i in range(feature_num):
    y, x, scale, orient = [float(_) for _ in lns[i*8+1].split()]

    cv2.circle(img, (int(x), int(y)), 2, GREEN, -1)

  cv2.imshow('_', img)
  k = cv2.waitKey(0)
  if k == 27: return


# Create a SIFT feature detector
sift = kornia.feature.SIFTFeature(
  num_features=2048, 
  upright=False, 
  rootsift=True
).to('cuda')

# Process an image
img_color = cv2.imread(IMAGE)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)


# 30 ms, for one (1000, 667) tensor
img_tensor = torch.from_numpy(img_gray[None, None]).to('cuda') / 255
with torch.inference_mode():
  lafs, responses, descs = sift(img_tensor)
keypoints = lafs[0, ..., -1 ]

for keypoint in keypoints:
  x, y = keypoint.cpu().numpy()
  cv2.circle(img_color, (int(x), int(y)), 2, GREEN, -1)

cv2.imshow('_', img_color)
cv2.waitKey(0)

print(lafs.shape)
print(responses.shape)
print(descs.shape)
print('DEBUG')

