import argparse
import os
import torch
from PKD.datasets.data import TestLoader
import cv2
from torchvision import utils
from tqdm import tqdm
from PIL import Image
from scipy.stats import pearsonr
import csv
import numpy as np
import os
import csv
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
from PKD.models.build_model import build_model

parser = argparse.ArgumentParser()

parser.add_argument('--img_size',default=(288,384), type=int)
parser.add_argument('--dataset_name',default='salicon', type=str)
parser.add_argument('--no_workers',default=8, type=int)
parser.add_argument('--results_dir',default="./result-val-SemiPKD", type=str)
parser.add_argument('--test_img_dir',default="/home/tercy/proj/datasets/saliency/saliency/salicon/stimuli/val", type=str)
parser.add_argument('--model_val_path',default="model_ensemble_ofa_eff4.pt", type=str)
parser.add_argument('--output_size', default=(480, 640))
parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--compute_cc',default=True, type=bool)

args = parser.parse_args()

output_viz = args.results_dir + "/" + "viz"

results_dir=os.path.join(output_viz)
os.makedirs(results_dir, exist_ok=True)

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
               normalize=True, range=None, scale_each=False, pad_value=0, format=None):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, value_range=range, scale_each=scale_each)

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
model_load_state_dict(subnet, args.model_val_path)
subnet = subnet.to(device)

os.makedirs(args.results_dir,exist_ok=True)

test_img_ids = os.listdir(args.test_img_dir)
test_dataset = TestLoader(args.test_img_dir, test_img_ids,arg=args)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

test_model(subnet, test_loader, device, args)

################## Gen CC to csv ##########################

def compute_correlation(val_path, gt_path, filename):
    val_img_ids = os.listdir(val_path)
    gt_img_ids = os.listdir(gt_path)

    val_img_ids = sorted(val_img_ids)
    gt_img_ids = sorted(gt_img_ids)

    # Create an empty list to store the results
    results = []

    for val_img_id, gt_img_id in zip(val_img_ids, gt_img_ids):
        # Process each pair of image IDs
        # Replace the following print statement with your desired code
        if val_img_id == gt_img_id:

            # Load and process the data for each image ID
            val_image = Image.open(os.path.join(val_path, val_img_id))
            gt_image = Image.open(os.path.join(gt_path, gt_img_id))

            # Convert the image data to a numpy array if needed
            val_data = np.array(val_image)
            gt_data = np.array(gt_image)

            val_data = val_data.flatten()
            gt_data = gt_data.flatten()

            # Compute the Pearson correlation coefficient and p-value
            correlation_coef, p_value = pearsonr(val_data, gt_data)

            # Store the results
            result = [correlation_coef, p_value, val_img_id]
            results.append(result)

    # Write the results to the CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Correlation Coefficient', 'p-value', 'filename'])  # Write the header
        writer.writerows(results)

    print("Results saved to", filename)
    

if args.compute_cc:
    val_path = args.results_dir
    gt_path = '/home/tercy/proj/datasets/saliency/saliency/salicon/saliency/val'
    compute_correlation(val_path, gt_path, output_viz+"/"+"cc_salnas_self_val.csv")
    
################## viusal for heat map ##########################

example_folder = sorted(os.listdir(args.test_img_dir))
results_folder = sorted(os.listdir(args.results_dir))

for i in tqdm(range(len(example_folder))):
    background = cv2.imread(os.path.join(args.test_img_dir, example_folder[i]))
    saliency = cv2.imread(os.path.join(args.results_dir, results_folder[i]))
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

    added_image = cv2.addWeighted(background,0.8, heatmap, 0.8, 0)
    basename_without_extension = os.path.splitext(os.path.basename(os.path.join(args.test_img_dir, example_folder[i])))[0]
    combined_output = output_viz + '/' + basename_without_extension +'_combined.jpg'
    cv2.imwrite(combined_output, added_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    
# 
# clean up
# 
def remove_png_files(directory):
    # Get the list of all files in the directory
    files = os.listdir(directory)
    
    # Iterate over each file
    for file in files:
        # Check if the file has the ".png" extension
        if file.endswith(".png"):
            # Construct the full file path
            file_path = os.path.join(directory, file)
            try:
                # Remove the file
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error while removing {file_path}: {e}")


# Specify the directory containing the PNG files
directory_path = args.results_dir

# Call the function to remove PNG files
remove_png_files(directory_path)
