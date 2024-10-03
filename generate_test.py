import argparse
import os
import torch
from PKD.datasets.data import TestLoader
import cv2, os
from torchvision import utils
from tqdm import tqdm
from PIL import Image
import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from PKD.models.build_model import build_model

parser = argparse.ArgumentParser()

parser.add_argument('--img_size',default=(288,384), type=int)
parser.add_argument('--dataset_name',default='salicon', type=str)
parser.add_argument('--no_workers',default=8, type=int)
parser.add_argument('--results_dir',default="./result-test", type=str)
parser.add_argument('--test_img_dir',default="/home/kongpasom/PKD/data/salicon/stimuli/test/", type=str)
parser.add_argument('--model_val_path',default="tmp.pt", type=str)
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--output_size', default=(480, 640))
parser.add_argument('--readout',default="simple", type=str)

args = parser.parse_args()

def test_model(model, loader, device, args):
    model.eval()
    with torch.no_grad():
        results_dir=os.path.join(args.results_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        for (img, img_id, sz) in tqdm(loader):
            img = img.to(device)
            
            pred_map = model(img)
            
            img=img.cpu().squeeze(0).clone().permute(1,2,0).numpy()
            pred_map = pred_map.cpu().squeeze(0).clone().numpy()
            pred_map = cv2.resize(pred_map, (int(sz[0]), int(sz[1])))
            
            pred_map = torch.FloatTensor(pred_map)
            img_save(pred_map, os.path.join(results_dir, img_id[0].replace('.jpg','.png')), normalize=True)

def img_save(tensor, fp, nrow=8, padding=2,
               normalize=True, scale_each=False, pad_value=0, format=None):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, scale_each=scale_each)

    ''' Add 0.5 after unnormalizing to [0, 255] to round to nearest integer '''
    
    ndarr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()[:,:,0]
    im = Image.fromarray(ndarr)
    exten = fp.split('.')[-1]
    if exten=="png":
        im.save(fp, format=format, compress_level=0)
    else:
        im.save(fp, format=format, quality=100) #for jpg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(args.model_val_path)

output_size = [480, 640]

def model_load_state_dict(student, path_state_dict):
    student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
    print("loaded pre-trained student and teacher")

subnet = build_model("eeeac2", args)
model_load_state_dict(subnet, "tmp.pt")
subnet = subnet.to(device)

os.makedirs(args.results_dir,exist_ok=True)

test_img_ids = os.listdir(args.test_img_dir)
test_dataset = TestLoader(args.test_img_dir, test_img_ids,arg=args)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

test_model(subnet, test_loader, device, args)
