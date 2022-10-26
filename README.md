# SwinDiD-supplementary
Supplementary Material for ``SwinDiD: Dimension-invariant Disentangling model with Swin Transformer for Light Field Super-Resolution''

In this supplemental material, we provide additional time complexity calculation details
and three additional comparison methods have been added to the main paper 
 
<a href="https://github.com/YuxinBao/YuxinBao.github.io/blob/main/SwinDiD-supp.pdf" target="_blank">PDF.</a>
![]()
## 1. Appendix A
A feature map of size $N \times N$ is fed to a convolution layer with a kernel $K \times K$ to output a feature map of size $M \times M$.
The corresponding time complexity of a single convolutional layer is formulated as


 <p align="center">$time \sim \mathcal{O}({M^2 \times K^2 \times C_{in} \times C_{out}})$ （1）</p>


where $C_{in}$ represents the number of input channels, that is, the number of output channels of previous layer. $C_{out}$ represents the number of convolution kernels in this convolutional layer, that is, 

the number of input channels of next layer.
It can be seen that the time complexity of each convolutional layer is completely determined by the area of the output feature map $M^2$, the area of the convolution kernel $K^2$, and the number of input $C_{in}$ and output channels $C_{out}$. 
The overall time complexity of a convolutional neural network is the sum of the time complexity of each convolutional layer. 


As shown in Figure \ref{png_2}, the input feature is of size $UH \times VW \times C$, where $U$ and $V$ are the angular resolution, $H$ and $W$ are the spatial resolutions. We define the angular resolution $U=V=A$, so the input feature of size can be express as $AH \times AW \times C$. 
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

In the convolution operation of the EFE-H, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to EFE-H with a kernel $1 \times A^2$ to output $F_{E\_H}$ of size $AH \times W \times C_{E\_H}$. The time complexity of EFE-H is $\mathcal{O}(\dfrac{1}{2}C^2A^3HW )$ with $C_{E\_H}= \dfrac{C}{2}$.
The feature $F_{E\_H}$ is fed into a $1\times1$ Conv operation to output $\hat{F}_{E\_H}$ of size $AH \times W \times \dfrac{AC}{2}$. 
The time complexity of $1\times1$ Conv operation is $\mathcal{O}(\dfrac{1}{4}C^2A^2HW )$ .

The time complexity of the EFE-H branch branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(\dfrac{1}{2}C^2A^3HW  +\dfrac{1}{4}C^2A^2HW )$  (4)</p>





In the convolution operation of the EFE-V, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to EFE-V with a kernel $A^2 \times 1$ to output $F_{E\_V}$ of size $H \times AW \times C_{E\_V}$. The time complexity of EFE-V is $\mathcal{O}(\dfrac{1}{2}C^2A^3HW )$ with $C_{E\_V}= \dfrac{C}{2}$.
The feature $F_{E\_V}$ is fed into a $1\times1$ Conv operation to output $\hat{F}_{E\_V}$ of size $H \times AW \times \dfrac{AC}{2}$. 
The time complexity of $1\times1$ Conv operation is $\mathcal{O}(\dfrac{1}{4}C^2A^2HW )$ .

The time complexity of the EFE-V branch branch of Distg-Block is calculated as follows:

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



In the convolution operation of the EFE-H, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to EFE-H with a kernel $1 \times A^2$ to output $F_{E\_H}$ of size $AH \times W \times C_{E\_H}$. The time complexity of EFE-H is $\mathcal{O}(CA^5HWY )$ with $C_{E\_H}= A^2Y$.

The time complexity of the EFE-H branch branch of Distg-Block is calculated as follows:

<p align="center">$\mathcal{O}(CA^5HWY )$  (10)</p>





In the convolution operation of the EFE-V, the input feature $F_{in}$ of size $AH \times AW \times C$ is fed to EFE-V with a kernel $A^2 \times 1$ to output $F_{E\_V}$ of size $H \times AW \times C_{E_V}$. The time complexity of EFE-V is $\mathcal{O}(CA^5HWY )$ with $C_{E_V}= A^2Y$.

The time complexity of the EFE-V branch branch of Distg-Block is calculated as follows:

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
On the basis of the main paper, we have added two SISR methods which are EDSR\cite{28}, and RCAN\cite{29} and a LF image SR method which is resLF \cite{30} for comparison, and compare the whole image with each other.

### 2.1 Quantitative Results
Table 1 shows the quantitative results achieved by SwinDiD in comparison with other state-of-the-art SR methods. SwinDiD achieves competitive PSNR and SSIM results on all five datasets of $\times 2 $ SR. 

### 2.2 Qualitative Results 
Figure \ref{2sr} and Figure \ref{4sr} show the whole image visual quality comparisons of different methods for $2 \times$ SR and $4 \times$ SR, respectively.
It can be seen that the SISR methods including VDSR\cite{27}, EDSR\cite{28}, and RCAN\cite{29}, fail to recover complex textures and details. In contrast, the deep learning based LF image SR methods can produce better visual effects than SISR methods, which is attribute to the use of different viewpoints information. However, edges and textures recovered by these methods are still suffer from blurring.
Compared with state-of-the-art methods, our SwinDiD can recover complex structures with shaper edges and fine details. Besides, we can observe that SwinDiD generates more appealing result than DiD in  $2 \times$ and $4 \times$ SR.
\begin{table*}
  \caption{PSNR/SSIM values achieved by different methods for $\times2$ and $\times 4$ SR. The best results are bolded.}
  \label{tab:commands}
   \resizebox{\textwidth}{!}
   {
  
  \begin{tabular}{ccccccc}
    \toprule
    \textrm{Method} & \textrm{scale} & \textrm{EPFL} & \textrm{HCInew} & \textrm{HCIold} & \textrm{INRIA} & \textrm{STFgantry} \\
    \midrule
   \textrm{Bicubic} &$\times 2$&29.74/0.938&31.89/0.936&37.69/0.979&31.33/0.958&31.06/0.950 \\

\textrm{VDSR \cite{27}}   &$\times 2$&32.50/0.974&34.37/0.956&40.61/0.987&34.44/0.974&35.54/0.979\\

\textrm{EDSR \cite{28}}  &$\times 2$&33.09/0.963&34.83/0.959&41.01/0.987&34.98/0.976&36.30/0.982\\

\textrm{RCAN \cite{29}}  &$\times 2$&33.16/0.963&35.02/0.960&41.13/0.987&35.04/0.977&36.67/0.983\\

\textrm{resLF \cite{30}}  &$\times 2$&33.62/0.971&36.69/0.974&43.42/0.993&35.36/0.980&38.35/0.990 \\

\textrm{LFSSR \cite{11}}  &$\times 2$&33.67/0.974&36.80/0.975&43.81/0.994&35.28/0.983&37.94/0.990\\

\textrm{LF-ATO \cite{8}}  &$\times 2$&34.27/0.976&37.24/0.977&44.21/0.994&36.17/0.984&39.64/0.993\\

\textrm{LF-InterNet \cite{12}}  &$\times 2$&34.11/0.976&37.17/0.976&44.57/\textbf{0.995}&35.83/0.984&38.43/0.991\\

\textrm{DistgSSR\cite{7}} &$\times 2$  &34.37/0.977 &37.73/0.979 &44.80/\textbf{0.995} &36.12/0.985&40.08/0.994\\

\textrm{DiD}  &$\times 2$&34.64/\textbf{0.978} &37.79/0.979 &\textbf{44.93}/\textbf{0.995} &36.53/0.985 &40.21/0.994 \\

\textrm{SwinDiD}  &$\times 2$&\textbf{34.71/0.978} &\textbf{37.97/0.980} &44.92/\textbf{0.995}&\textbf{36.62/0.986}& \textbf{40.70/0.995}\\

\midrule
\textrm{Bicubic} &$\times 4$&25.26/0.832&27.71/0.852&32.58/0.934&26.95/0.887&26.09/0.854\\

\textrm{VDSR \cite{27}} &$\times 4$&27.25/0.878&29.31/0.882&34.81/0.952&29.19/0.920&28.51/0.901\\

\textrm{EDSR \cite{28}} &$\times 4$&27.83/0.885&29.59/0.887&35.18/0.954&29.65/0.926&28.70/0.907\\

\textrm{RCAN \cite{29}}  &$\times 4$&27.90/0.886&29.69/0.889&35.36/0.955&29.80/0.928&29.02/0.913\\

\textrm{resLF \cite{30}} &$\times 4$&28.26/0.904&30.72/0.911&36.71/0.968&30.34/0.941&30.19/0.937\\

\textrm{LFSSR \cite{11}} &$\times 4$&28.59/0.912&30.93/0.914&36.90/0.970&30.58/0.947&30.57/0.943\\

\textrm{LF-ATO \cite{8}} &$\times 4$&28.51/0.912&30.88/0.913&37.00/0.970&30.71/0.948&30.61/0.943\\

\textrm{LF-InterNet \cite{12}} &$\times 4$&28.51/0.912&30.82/0.914&36.83/0.970&30.50/0.947&30.22/0.938\\

\textrm{DistgSSR \cite{7}} &$\times 4$&{28.77}/0.915&31.16/0.918&37.25/0.972&30.82/0.949&31.03/0.949\\

\textrm{DiD}  &$\times 4$&\textbf{28.86}/0.917&31.29/0.920&37.48/\textbf{0.973}&\textbf{31.04}/\textbf{0.951}&31.26/0.951\\

\textrm{SwinDiD} &$\times 4$&\textbf{28.86}/\textbf{0.918}&\textbf{31.30}/\textbf{0.921}&\textbf{37.52}/\textbf{0.973}&30.93/\textbf{0.951}&\textbf{31.37}/\textbf{0.952}\\
    \bottomrule
  \end{tabular}
  }
\end{table*}




\begin{figure*}[htbp]
  \centering
  % Requires \usepackage{graphicx}
  \includegraphics[width =\textwidth]{2sr-all.png}
  \caption{ Visual quality comparisons of  $2 \times $ SR for different methods.}\label{2sr}
\end{figure*}


\begin{figure*}[htbp]
  \centering
  % Requires \usepackage{graphicx}
  \includegraphics[width =\textwidth]{4sr-all.png}
  \caption{ Visual quality comparisons of $4 \times$ SR for different methods.}\label{4sr}
\end{figure*}




\clearpage
\bibliographystyle{IEEEbib}
\bibliography{strings,conference-suoxie}

\end{document}
\endinput
