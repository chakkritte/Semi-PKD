import torch
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
from PKD.models.build_model import build_model
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_dir", type=str, default="/home/kongpasom/PKD/data/")
parser.add_argument('--input_size_h',default=384, type=int)
parser.add_argument('--input_size_w',default=384, type=int)
parser.add_argument('--no_workers',default=12, type=int)
parser.add_argument('--no_epochs',default=20, type=int)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--lr_sched',default=True, type=bool)
parser.add_argument('--model_val_path',default="model.pt", type=str)
parser.add_argument('--model_salicon_path',default="model_efb4.pt", type=str)
parser.add_argument('--output_dir', type=str, default="outputs")

parser.add_argument('--kldiv', action='store_true', default=True)
parser.add_argument('--cc', action='store_true', default=True)
parser.add_argument('--nss', action='store_true', default=True)
parser.add_argument('--sim', action='store_true', default=False)
parser.add_argument('--l1', action='store_true', default=False)
parser.add_argument('--auc', action='store_true', default=False)

parser.add_argument('--dataset',default="salicon", type=str)
parser.add_argument('--teacher',default="efb4", type=str)
parser.add_argument('--student',default="eeeac2", type=str)
parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))

parser.add_argument('--amp',default=True, type=bool)
parser.add_argument('--seed',default=3407, type=int)

args = parser.parse_args()

img = Image.open("000000374003.jpg")

gt_size = [288, 384]
output_size = (480, 640)
# mean = [0.5, 0.5, 0.5]
# std  = [0.5, 0.5, 0.5]

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform_pipeline = transforms.Compose([
    transforms.Resize((gt_size[0], gt_size[1])),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])    

img = transform_pipeline(img)

img = img.unsqueeze(0)
img = Variable(img).cuda()

def model_load_state_dict(student , teacher, path_state_dict):
    student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
    teacher.load_state_dict(torch.load(path_state_dict)["teacher"], strict=True)
    print("loaded pre-trained student and teacher")

student = build_model(args.student, args)
teacher = build_model(args.teacher, args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher.to(device)
student.to(device)

model_load_state_dict(student , teacher, "model-c2-ccun.pt")

teacher.eval()
student.eval()

prediction_t = teacher(img)
prediction_s = student(img)

from torchvision.utils import save_image

save_image(prediction_t, 'teacher.png')
save_image(prediction_s, 'student.png')

print(prediction_t.size())

#################### visual color ##############

import cv2
#input
background = cv2.imread('000000374003.jpg')
#saliency prediction
saliency = cv2.imread('teacher.png', 0)
heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

added_image = cv2.addWeighted(background,0.8, heatmap, 0.8, 0)

cv2.imwrite('combined.jpg', added_image)
