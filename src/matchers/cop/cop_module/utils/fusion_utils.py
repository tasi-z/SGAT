import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import numpy as np

def extract_frequency_components_batch(images):
    # images: (batch_size, channels, height, width)
    batch_size, channels, height, width = images.size()
    
    # Prepare mask for low frequency
    crow, ccol = height // 2, width // 2
    r = 30  # Radius for low frequency mask
    mask = np.zeros((height, width), np.uint8)
    mask[crow-r:crow+r, ccol-r:ccol+r] = 1
    mask = torch.tensor(mask, dtype=torch.float32, device=images.device)
    
    # Initialize outputs
    low_freq_images = torch.zeros_like(images)
    high_freq_images = torch.zeros_like(images)
    
    for i in range(batch_size):
        for c in range(channels):
            # Get the channel
            img_channel = images[i, c, :, :]
            
            # Compute the 2D FFT
            f = torch.fft.fft2(img_channel)
            fshift = torch.fft.fftshift(f)
            
            # Apply mask for low frequency
            fshift_low = fshift * mask
            img_low = torch.fft.ifft2(torch.fft.ifftshift(fshift_low)).real
            
            # Apply inverse mask for high frequency
            fshift_high = fshift * (1 - mask)
            img_high = torch.fft.ifft2(torch.fft.ifftshift(fshift_high)).real
            
            # Store results
            low_freq_images[i, c, :, :] = img_low
            high_freq_images[i, c, :, :] = img_high
    
    return low_freq_images, high_freq_images

def extract_frequency_components(img):
    # Compute the 2D Fast Fourier Transform (FFT) of the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Create a mask for low frequencies
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r = 30  # Radius for low frequency mask
    mask[crow-r:crow+r, ccol-r:ccol+r] = 1
    
    # Apply the mask to get the low frequency part
    fshift_low = fshift * mask
    img_low = np.fft.ifft2(np.fft.ifftshift(fshift_low))
    img_low = np.abs(img_low)
    
    # Apply the inverse mask to get the high frequency part
    fshift_high = fshift * (1 - mask)
    img_high = np.fft.ifft2(np.fft.ifftshift(fshift_high))
    img_high = np.abs(img_high)
    
    return img_low, img_high, magnitude_spectrum

# def get_cosine_similarity(input1, input2, epsilon=1e-8):
#     # Flatten the spatial dimensions
#     batch_size, channels, height, width = input1.shape
#     input1 = input1.view(batch_size, -1)  # Flatten to (batch_size, feature_dim)
#     input2 = input2.view(batch_size, -1)

#     # L2 normalize
#     norm_A = torch.norm(input1, dim=1, keepdim=True) + epsilon
#     norm_B = torch.norm(input2, dim=1, keepdim=True) + epsilon
#     input1_normalized = input1 / norm_A
#     input2_normalized = input2 / norm_B

#     # Compute cosine similarity for each sample in the batch
#     cosine_similarity = torch.sum(input1_normalized * input2_normalized, dim=1)

#     return cosine_similarity.mean()
def get_cosine_similarity(input1, input2, epsilon=1e-8):
    # Flatten the spatial dimensions
    batch_size, channels, height, width = input1.shape
    input1=input1.permute(0,2,3,1).reshape(batch_size,-1,channels)
    input2=input2.permute(0,2,3,1).reshape(batch_size,-1,channels)

    #输入调整为（batch_size,num,channels）
    #0和0的余弦相似性是0
    cosine_similarity = F.cosine_similarity(input1, input2, dim=-1, eps=1e-8)
    # # 保持输出维度为 [batch_size]
    # mean_cosine_similarity = cosine_similarity.mean(dim=1)
    # 保持输出维度为 []
    mean_cosine_similarity = cosine_similarity.mean()
    return mean_cosine_similarity
def cal_orth(x,y):
    #计算x和y的正交向量，得到的向量与x正交
    # x=x.permute(0,2,1)
    # y=y.permute(0,2,1)
    res=torch.zeros_like(x)        
    for i in range(x.shape[0]):#对于每个batch
        proj = torch.matmul(x[i], y[i].t())
        x_norm = torch.norm(x[i], p=2, dim=1, keepdim=True)
        y_norm = torch.norm(y[i], p=2, dim=1, keepdim=True)
        proj=torch.diag(proj)#diag是对角线元素
        proj_A=proj/(x_norm**2).t()#
        proj_A=x[i]*proj_A.reshape(-1,1)#y在x上的投影
        vertical_x_y=y[i]-proj_A #正交
        res[i]=vertical_x_y
    # res=res.permute(0,2,1)
    return res