
import numpy as np
import torch
from leaptorch import Projector
from matplotlib import pyplot as plt

device_name = "cuda:0"
device = torch.device(device_name)
proj = Projector(use_gpu=True, gpu_device=device)

dim_x = 512
dim_y = 512
dim_z = 64
voxel_width =  0.8
voxel_height = 0.8
center_offset_x = 0.0
center_offset_y = 0.0
center_offset_z = 280.0
# center_offset_z = voxel_height*(dim_z-1.0)/2.0 
# center_offset_z = 300.0
# center_offset_z = 0.0
proj.set_volume(dim_x, 
                dim_y, 
                dim_z, 
                voxel_width, 
                voxel_height, 
                center_offset_x, 
                center_offset_y, 
                center_offset_z)

nangles = 288
nrows = 1024
nrows = 1536
ncols = 1024
ncols = 2048
pixel_height = 0.8
pixel_width = 0.8
# center_row = (nrows-1.0)/2.0 - center_offset_z/pixel_height
center_row = (nrows-1.0)/2.0
center_col = (ncols-1.0)/2.0
# center_row = 1000.0
# center_col = (ncols-1.0)/2.0


arange = 360
phis = torch.linspace(0, arange - arange/nangles, nangles, device='cpu')

cone_beam=True
if cone_beam:
    sod = 500.0
    sdd = 1000.0
    proj.set_cone_beam(nangles, 
                    nrows, 
                    ncols, 
                    pixel_height, 
                    pixel_width, 
                    center_row, 
                    center_col, 
                    arange, 
                    phis, 
                    sod, 
                    sdd)
    
else:
    proj.set_parallel_beam( nangles, 
                            nrows, 
                            ncols, 
                            pixel_height, 
                            pixel_width, 
                            center_row, 
                            center_col, 
                            arange, 
                            phis)
    


def forward_project(volume):
    projections = proj.forward(volume.unsqueeze(0))[0]
    return projections

def back_project(projections):
    proj.forward_project = False
    volume = proj.forward(projections.unsqueeze(0))[0]
    proj.forward_project = True
    return volume


volume_ones = torch.ones(dim_z, dim_y, dim_x, device=device, dtype=torch.float32)
projections_ones = torch.ones(nangles, nrows, ncols, device=device, dtype=torch.float32)

fp_volume_ones = forward_project(volume_ones)

plt.figure()
# plt.imshow(fp_volume_ones[:,nrows//2,:].cpu().numpy())
plt.imshow(fp_volume_ones[0].cpu().numpy())
plt.colorbar()
plt.savefig("figures/fp_volume_ones.png")

bp_projections_ones = back_project(projections_ones)

plt.figure()
plt.imshow(bp_projections_ones[dim_z//2].cpu().numpy())
plt.colorbar()
plt.savefig("figures/bp_projections_ones.png")