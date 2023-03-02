# DiD-supplementary
Supplementary Material for ``DiD: Dimension-invariant Disentangling model for Light Field Super-Resolution''

In this supplementary material, we provide additional time complexity calculation details
and three additional comparison methods have been added to the main paper. 
 


 

## 1. Appendix A
A feature map of size $N \times N$ is fed to a convolution layer with a kernel $K \times K$ to output a feature map of size $M \times M$.
The corresponding time complexity of a single convolutional layer is formulated as


 <p align="center">$time \sim \mathcal{O}({M^2 \times K^2 \times C_{in} \times C_{out}})$ （1）</p>


where $C_{in}$ represents the number of input channels, that is, the number of output channels of previous layer. $C_{out}$ represents the number of convolution kernels in this convolutional layer, that is the number of input channels of next layer.
It can be seen that the time complexity of each convolutional layer is completely determined by the area of the output feature map $M^2$, the area of the convolution kernel $K^2$, and the number of input $C_{in}$ and output channels $C_{out}$. 
The overall time complexity of a convolutional neural network is the sum of the time complexity of each convolutional layer. 

<p align="center"><img src="https://github.com/YuxinBao/DiD/blob/main/Block.png" width="600px"></p>
 <p align="center">Fig.1. Illustration of one Distg-Block and DiD-Block.</p>
 
 
As shown in Fig.1, the input feature is of size $UH \times VW \times C$, where $U$ and $V$ are the angular resolution, $H$ and $W$ are the spatial resolutions. We define the angular resolution $U=V=A$, so the input feature of size can be expressed as $AH \times AW \times C$. 
Pixel shuffle and Concatenate operations have less time complexity, therefore it is not computed in the calculation of the time complexity. Next, we will calculate the time complexity of Distg-Block and DiD-Block respectively.

### 1.1 Time Complexity of Distg-Block

In the convolution operation of the AFE, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to AFE with a kernel $A \times A$ to output $F_A$ of size $H \times W \times C_A$. The time complexity of AFE is $\mathcal{O}(\dfrac{1}{4}C^2A^2HW )$ with $C_A= \dfrac{C}{4}$.
The feature $F_A$ is fed into a $1\times1$ Conv operation to output $\hat{F}_A$ of size $H \times W \times \dfrac{A^2C}{4}$. 
The time complexity of $1\times1$ Conv operation is $\mathcal{O}(\dfrac{1}{16}C^2A^2HW )$ .


The time complexity of the angular feature extraction branch of Distg-Block is calculated as follows:


<p align="center">$\mathcal{O}(\dfrac{1}{4}C^2A^2HW  + \dfrac{1}{16}C^2A^2HW)$  (2)</p>




In the convolution operation of the SFE, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to SFE with a kernel $3 \times 3$ to output $F_S$ of size $AH \times AW \times C_S$. The time complexity of SFE is $\mathcal{O}(9C^2A^2HW )$ with $C_S= C$.
The feature $F_S$ is fed into another SFE with a kernel $3 \times 3$ to output $F'_S$ of size $AH \times AW \times C'_S$. 
The time complexity of this SFE is $\mathcal{O}(9C^2A^2HW )$ with $C'_S= C$.

The time complexity of the spatial feature extraction branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(9C^2A^2HW +9C^2A^2HW )$  (3)</p>

In the convolution operation of the EFE-H, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to EFE-H with a kernel $1 \times A^2$ to output $F_{E-H}$ of size $AH \times W \times C_{E-H}$. The time complexity of EFE-H is $\mathcal{O}(\dfrac{1}{2}C^2A^3HW )$ with $C_{E-H}= \dfrac{C}{2}$.
The feature $F_{E-H}$ is fed into a $1\times1$ Conv operation to output $\hat{F}_{E-H}$ of size $AH \times W \times \dfrac{AC}{2}$. 
The time complexity of $1\times1$ Conv operation is $\mathcal{O}(\dfrac{1}{4}C^2A^2HW )$ .

The time complexity of the EFE-H branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(\dfrac{1}{2}C^2A^3HW  +\dfrac{1}{4}C^2A^2HW )$  (4)</p>





In the convolution operation of the EFE-V, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to EFE-V with a kernel $A^2 \times 1$ to output $F_{E-V}$ of size $H \times AW \times C_{E-V}$. The time complexity of EFE-V is $\mathcal{O}(\dfrac{1}{2}C^2A^3HW )$ with $C_{E-V}= \dfrac{C}{2}$.
The feature $F_{E-V}$ is fed into a $1\times1$ Conv operation to output $\hat{F}_{E-V}$ of size $H \times AW \times \dfrac{AC}{2}$. 
The time complexity of $1\times1$ Conv operation is $\mathcal{O}(\dfrac{1}{4}C^2A^2HW )$ .

The time complexity of the EFE-V branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(\dfrac{1}{2}C^2A^3HW  +\dfrac{1}{4}C^2A^2HW )$  (5)</p>



After the pixel shuffle and feature concatenate, the feature becomes $F_c = AH \times AW \times \dfrac{9}{4}C$ . The feature $F_c$ is fed into a $1\times1$ Conv operation to output $\hat{F}_c$ of size $AH \times AW \times C$. The time complexity of $1 \times 1$ Conv is $\mathcal{O}(\dfrac{9}{4}C^2A^2HW )$.
The feature $\hat{F}_c$ is fed into SFE to output $F$ of size $AH \times AW \times C$. The time complexity of SFE is $\mathcal{O}(9C^2A^2HW )$.
The time complexity of this two operation in Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(\dfrac{9}{4}C^2A^2HW + 9C^2A^2HW )$  (6)</p>


The time complexity of Distg-Block is the sum of Eq.(2) to Eq.(6):

<p align="center"> $time_{Distg}=\mathcal{O}(C^2A^3HW + 30.0625C^2A^2HW )$  (7)</p>





### 1.2 Time Complexity of DiD-Block
In the convolution operation of the AFE, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to AFE with a kernel $A \times A$ to output $F_A$ of size $H \times W \times C_A$. The time complexity of AFE is $\mathcal{O}(CA^4HWY)$ with $C_A= A^2Y$.

The time complexity of the angular feature extraction branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(CA^4HWY )$  (8)</p>




In the convolution operation of the SFE, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to SFE with a kernel $3 \times 3$ to output $F_S$ of size $AH \times AW \times C_S$. The time complexity of SFE is $\mathcal{O}(9C^2A^2HW )$ with $C_S= C$.
The feature $F_S$ is fed into another SFE with a kernel $3 \times 3$ to output $F'_S$ of size $AH \times AW \times C'_S$. 
The time complexity of this SFE is $\mathcal{O}(9CA^4HWY )$ with $C'_S= A^2Y$ .


The time complexity of the spatial feature extraction branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(9C^2A^2HW +9CA^4HWY)$  (9)</p>



In the convolution operation of the EFE-H, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to EFE-H with a kernel $1 \times A^2$ to output $F_{E-H}$ of size $AH \times W \times C_{E-H}$. The time complexity of EFE-H is $\mathcal{O}(CA^5HWY )$ with $C_{E-H}= A^2Y$.

The time complexity of the EFE-H branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(CA^5HWY )$  (10)</p>





In the convolution operation of the EFE-V, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to EFE-V with a kernel $A^2 \times 1$ to output $F_{E-V}$ of size $H \times AW \times C_{E-V}$. The time complexity of EFE-V is $\mathcal{O}(CA^5HWY )$ with $C_{E-V}= A^2Y$.

The time complexity of the EFE-V branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(CA^5HWY)$  (11)</p>




After the pixel shuffle and feature concatenate, the feature becomes $F_c = AH \times AW \times C$ . 
The feature $F_c$ is fed into SFE with a kernel $3 \times 3$ to output $F$ of size $AH \times AW \times C$. The time complexity of SFE is as follows:

<p align="center">$\mathcal{O}(9C^2A^2HW)$  (12)</p>

The time complexity of DiD-Block is the sum of Eq.(8) to Eq.(12):


 <p align="center">$time_{DiD} = \mathcal{O}((18C^2A^2HW+ 10CA^4HWY + 2CA^5HWY)$  (13)</p>


from the formulas $C = (A+1)^2Y$ in the main paper,  Eq.(13) can becomes:

<p align="center">$time_{DiD} = \mathcal{O}((22 + 2A +\dfrac{2A^2-10A-4}{(A+1)^2})C^2A^2HW)$  (14)</p>


Therefore, according to Eq.(7) and Eq.(14), the time complexity of Distg-Block and DiD-Block are $\mathcal{O}((30+A)C^2A^2HW)$ and  $\mathcal{O}((22+2A)C^2A^2HW)$, respectively. 


## 2. Appendix B
On the basis of the main paper, we have added two SISR methods which are EDSR [2], and RCAN [3] and a LF image SR method which is resLF [4] for comparison, and compare the whole image with each other.

### 2.1 Quantitative Results
Table 1 and Table 2 show the quantitative results achieved by SwinDiD in comparison with other state-of-the-art SR methods on 2 × and 4 × SR, respectively. SwinDiD achieves competitive PSNR and SSIM results on all five datasets on 2 × and 4 × SR. 

Table 1. PSNR/SSIM values achieved by different methods for 2 × SR. The best results are bolded.
 
| Method | scale | EPFL |HCInew | HCIold | INRIA | STFgantry|
| :----:| :----: | :----: | :----: | :----:| :----: | :----: |
| Bicubic |  2|  29.74/0.938|31.89/0.936|37.69/0.979|31.33/0.958|31.06/0.950|
| VDSR [1]| 2 | 32.50/0.974|34.37/0.956|40.61/0.987|34.44/0.974|35.54/0.979|
| EDSR [2]|  2| 33.09/0.963|34.83/0.959|41.01/0.987|34.98/0.976|36.30/0.982|
| RCAN [3]|  2| 33.16/0.963|35.02/0.960|41.13/0.987|35.04/0.977|36.67/0.983|
| resLF [4]| 2 | 33.62/0.971|36.69/0.974|43.42/0.993|35.36/0.980|38.35/0.990|
| LFSSR [5]| 2 | 33.67/0.974|36.80/0.975|43.81/0.994|35.28/0.983|37.94/0.990|
| LF-ATO [6]| 2 | 34.27/0.976|37.24/0.977|44.21/0.994|36.17/0.984|39.64/0.993|
| LF-InterNet [7]| 2 | 34.11/0.976|37.17/0.976|44.57/**0.995**|35.83/0.984|38.43/0.991|
| DistgSSR [8]| 2 | 34.37/0.977 |37.73/0.979 |44.80/**0.995** |36.12/0.985|40.08/0.994|
| DiD | 2 |34.64/0.978 |37.79/0.979 |44.93/**0.995** |36.53/0.985 |40.21/0.994|
| DiD-SwinT | 2 |34.71/0.978|37.97/0.980 |44.92/**0.995**|36.62/**0.986**| 40.70/**0.995**|
| DiD-MST++ | 2 |**35.03/0.979**|**38.09/0.980** |**45.08**/**0.995**|**36.92/0.986**| **40.75/0.995**|

Table 2. PSNR/SSIM values achieved by different methods for 4 × SR. The best results are bolded.
| Method | scale | EPFL |HCInew | HCIold | INRIA | STFgantry|
| :----:| :----: | :----: | :----: | :----:| :----: | :----: |
| Bicubic |  4|  25.26/0.832|27.71/0.852|32.58/0.934|26.95/0.887|26.09/0.854|
| VDSR [1]| 4| 27.25/0.878|29.31/0.882|34.81/0.952|29.19/0.920|28.51/0.901|
| EDSR [2]|  4| 27.83/0.885|29.59/0.887|35.18/0.954|29.65/0.926|28.70/0.907|
| RCAN [3]|  4| 27.90/0.886|29.69/0.889|35.36/0.955|29.80/0.928|29.02/0.913|
| resLF [4]| 4 |  28.26/0.904|30.72/0.911|36.71/0.968|30.34/0.941|30.19/0.937|
| LFSSR [5]| 4 | 28.59/0.912|30.93/0.914|36.90/0.970|30.58/0.947|30.57/0.943|
| LF-ATO [6]| 4 | 28.51/0.912|30.88/0.913|37.00/0.970|30.71/0.948|30.61/0.943|
| LF-InterNet [7]| 4 | 28.51/0.912|30.82/0.914|36.83/0.970|30.50/0.947|30.22/0.938|
| DistgSSR [8]| 4 |28.77/0.915|31.16/0.918|37.25/0.972|30.82/0.949|31.03/0.949|
| DiD | 4 | 28.74/0.916|31.21/0.919|37.31/0.972|30.86/0.950|31.09/0.949|
| DiD-SwinT | 4 | 28.71/0.914|31.17/0.919|37.17/0.971|30.83/0.950|31.11/0.949|
| DiD-MST++ | 4 | **28.92/0.917**|**31.35/0.921**|**37.50/0.973**|31.02/**0.952**|**31.27/0.952**|


<!---
### 2.2 Qualitative Results 

Fig.2 and Fig.3 show the whole image visual quality comparisons of different methods for 2 × and 4 × SR, respectively.
It can be seen that the SISR methods including VDSR [2], EDSR [3] and RCAN [4], fail to recover complex textures and details. In contrast, the deep learning based LF image SR methods can produce better visual effects than SISR methods, which is attribute to the use of different viewpoints information. However, edges and textures recovered by these methods are still suffer from blurring.
Compared with state-of-the-art methods, our SwinDiD can recover complex structures with shaper edges and fine details. Besides, we can observe that SwinDiD generates more appealing result than DiD in  2 × and 4 × SR.

<p align="center"><img src="https://github.com/YuxinBao/SwinDiD/blob/main/2×SR.png" width="600px"></p>
 <p align="center">Fig.2. Visual quality comparisons of 2 × SR for different methods.</p>
 
 <p align="center"><img src="https://github.com/YuxinBao/SwinDiD/blob/main/4×SR.png" width="600px"></p>
 <p align="center">Fig.3. Visual quality comparisons of  4 × SR for different methods.</p>
-->





### Reference

[1] J. K. Lee J. Kim and K. M. Lee, “Accurate image super-resolution using very deep convolutional networks,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 1646–1654.

[2] B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee, “Enhanced deep residual networks for single image superresolution,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW),2017, pp. 1132–1140.

[3] Y. Zhang, K. Li, K. Li, L. Wang, B. Zhong, and Y. Fu, “Image super-resolution using very deep residual channel attention networks,” 2018.

[4] S. Zhang, Y. Lin, and H. Sheng, “Residual networks for light field image super-resolution,” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11038–11047, 2019.

[5] H. W. F. Yeung, X. Chen J. Hou, J. Chen, Z. Chen, and Y. Y. Chung, “Light field spatial super-resolution using deep efficient spatial-angular separable convolution,” IEEE Transactions on Image Processing, vol. 28, no. 5, pp. 2319–2330, 2019.

[6] J. Jin, J. Hou, J. Chen, and S. Kwong, “Light field spatial super-resolution via deep combinatorial geometry embedding and structural consistency regularization,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 2257–2266.

[7] Y. Wang, L. Wang, J. Yang, W. An, J. Yu, and Y. Guo, “Spatial-angular interaction for light field image super-resolution,” ArXiv, vol. abs/1912.07849, 2020.

[8] Y. Wang, L. Wang, G. Wu, J. Yang, W. An, J. Yu, and Y. Guo, “Disentangling light fields for super-resolution and disparity estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–1, 2022.


