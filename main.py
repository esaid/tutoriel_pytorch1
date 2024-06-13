from torchvision import models
import torch

print(dir(models))
alexnet = models.alexnet()
resnet = models.resnet101(pretrained=True)
resnet
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
import urllib.request
from PIL import Image

lhassa = 'https://www.photos-animaux.com/data/photos/518/5178/517790.jpg'
bobby = "https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch2/bobby.jpg"
monimage = lhassa
urllib.request.urlretrieve(monimage, "chien.jpg")
img = Image.open("chien.jpg")
img.show()
img_t = preprocess(img)
img_t

batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)
# print(out)
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch2/imagenet_classes.txt',
    "data.txt")
with open("data.txt") as f:
    labels = [line.strip() for line in f.readlines()]
    print(labels)

    _, index = torch.max(out, 1)
print(index)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())
_, indices = torch.sort(out, descending=True)
[print(labels[idx], percentage[idx].item()) for idx in indices[0][:3]]
