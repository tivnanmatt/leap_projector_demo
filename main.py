
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
center_offset_z = (dim_z-1.0)/2.0 + 200
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
ncols = 1024
pixel_height = 0.8
pixel_width = 0.8
center_row = (nrows-1.0)/2.0
center_col = (ncols-1.0)/2.0

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
    


# load TCGA_LIHC_000401.npy
volume_true = np.load("data/TCGA_LIHC_000401.npy")[10:74]
mu_H2O_80kev = 1.837E01 # mm^2/g
volume_true = ((volume_true + 1000) / 1000) * mu_H2O_80kev
volume_true = torch.tensor(volume_true, dtype=torch.float32, device=device)

projections_true = proj.forward(volume_true.unsqueeze(0))[0].clone()

plt.figure()
plt.imshow(volume_true[dim_z//2, :, :].cpu().numpy())
plt.colorbar()
plt.savefig("figures/volume_true.png")
plt.close()

plt.figure()
if nangles == 1:
    plt.plot(projections_true[0, 650, :].cpu().numpy())
else:
    plt.imshow(projections_true[:, 650, :].cpu().numpy())
    plt.colorbar()
plt.savefig("figures/projections_true.png")


def forward_project(volume):
    projections = proj.forward(volume.unsqueeze(0))[0]
    return projections

def back_project(projections):
    proj.forward_project = False
    volume = proj.forward(projections.unsqueeze(0))[0]
    proj.forward_project = True
    return volume

def preconditioned_gradient_descent(projections, 
                                    volume_pred_init, 
                                    step_size=1e-2,
                                    inverse_hessian_approximator=None,
                                    niter=1000):
    
    volume_pred = volume_pred_init.clone()

    if inverse_hessian_approximator is None:
        def inverse_hessian_approximator(gradient):
            return gradient
        
    for i in range(niter):
        projections_pred = forward_project(volume_pred)
        gradient = back_project(projections - projections_pred)
        volume_pred = volume_pred + step_size*inverse_hessian_approximator(gradient.unsqueeze(0))[0]
        rmse = torch.sqrt((volume_true - volume_pred)**2).mean()
        print("Iteration: %d,, RMSE: %f" % (i,  rmse.item()))

    projections_pred = forward_project(volume_pred)

    return volume_pred, projections_pred


def ramp_filter(    volume_input,
                    power=None,
                    dc_offset=None,
                    device=device):
    
    dim_x = volume_input.shape[3]
    dim_y = volume_input.shape[2]
    dim_z = volume_input.shape[1]
    batch_size = volume_input.shape[0]

    volume_output = torch.zeros_like(volume_input)
    volume_input_fft = torch.fft.fftn(volume_input, dim=(3, 2))

    freq_x = np.fft.fftfreq(dim_x)
    freq_y = np.fft.fftfreq(dim_y)
    yGrid, xGrid = torch.meshgrid(torch.tensor(freq_y), torch.tensor(freq_x))
    rGrid = torch.sqrt(xGrid**2 + yGrid**2)
    rGrid = rGrid.unsqueeze(0).unsqueeze(0).repeat(batch_size, dim_z, 1, 1)
    rGrid.to(torch.float32).to(device)

    ramp_filter = torch.abs(rGrid)
    if dc_offset is None:
        dc_offset = torch.min(ramp_filter[ramp_filter>0]).item()
    ramp_filter[ramp_filter==0] = dc_offset

    if power is not None:
        ramp_filter = ramp_filter**power

    ramp_filter = ramp_filter.to(torch.float32).to(device)
    volume_output_fft = volume_input_fft * ramp_filter
    volume_output = torch.fft.ifftn(volume_output_fft, dim=(3, 2)).real

    return volume_output

class RampFilter(torch.nn.Module):
    def __init__(self, voxel_width, voxel_height, power, dc_offset=None):
        super(RampFilter, self).__init__()
        self.voxel_width = voxel_width
        self.voxel_height = voxel_height
        self.power = power
        self.dc_offset = dc_offset
    def forward(self, volume_input):
        return ramp_filter(volume_input, power=self.power, dc_offset=self.dc_offset)

class Identity(torch.nn.Module):
    def __init__(self, voxel_width, voxel_height):
        super(Identity, self).__init__()
        self.voxel_width = voxel_width
        self.voxel_height = voxel_height
    def forward(self, volume_input):
        return volume_input
    

volume_pred = torch.zeros_like(volume_true)

# run 100 iterations of ART to get a rough estimate of the image
volume_pred, projections_pred = preconditioned_gradient_descent(projections_true, 
                                                                volume_pred, 
                                                                inverse_hessian_approximator=Identity(voxel_width, voxel_height),
                                                                step_size=1e-3/nangles,
                                                                niter=100)

# run 100 iterations of preconditioned gradient descent using the ramp filter as the inverse hessian approximator
volume_pred, projections_pred = preconditioned_gradient_descent(projections_true, 
                                                                volume_pred, 
                                                                inverse_hessian_approximator=RampFilter(voxel_width, voxel_height, power=1.0), 
                                                                step_size=5e-3/nangles,
                                                                niter=100)


plt.figure()
plt.imshow(volume_pred[dim_z//2, :, :].cpu().numpy(), vmin=torch.min(volume_true[dim_z//2, :, :]).cpu().numpy(), vmax=torch.max(volume_true[dim_z//2, :, :]).cpu().numpy())
plt.colorbar()
plt.savefig("figures/volume_pred.png")

plt.figure()
if nangles == 1:
    plt.plot(projections_pred[0, 650, :].cpu().numpy())
else:
    plt.imshow(projections_pred[:, 650, :].cpu().numpy())
    plt.colorbar()
plt.savefig("figures/projections_pred.png")

