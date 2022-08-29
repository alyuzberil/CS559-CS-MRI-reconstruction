# CS559 DEEP LEARNING PROJECT: ISTA-Net+ with two priors 

Implementation and extension of ISTA-Net+ architecture such that two different sparsifying transformations are learned to solve the ill-conditioned inverse problem better and improve the reconstruction quality in Compressed Sensing MRI (CS-MRI)

Authors:

- Beril Alyüz
- Aslı Alpman

CS-MRI aims to reduce the scan time of MRI by using alternative sampling approaches which exploit the redundancies in k-space data. Then, the ill-conditioned inverse problem is solved by enforcing sparsity on some hand-crafted transform domain with iterative complex optimization algorithms to reconstruct the image. It is recently shown that deep learning methods can learn more versatile transformations, which significantly improves the reconstruction results. ISTA-Net+ unrolls the ISTA  (Iterative Shrinkage-Thresholding Algorithm) iterations and uses deep learning techniques to learn sparsifying transformation[1].  This project aims to extend ISTA-Net+ architecture, which uses deep learning to learn a sparsifying transformation, to learn two different sparsifying transformations. In this way, we obtained two distinct prior and more information which improved the reconstruction quality. 

The implementation is based on the ISTA-Net+ Pytorch GitHub repository provided by the authors of the paper[2]. The dataset can also be found at the same repository.

Train_MRI_CS_ISTA_Net_plus.py --> Train script

Testing can be done with the test script "TEST_MRI_CS_ISTA_Net_plus.py" at the source GitHub repo. 

References
[1] J. Zhang and B. Ghanem, “ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.
 
[2] Jianzhangcs, “jianzhangcs/ISTA-Net-PyTorch,” GitHub. [Online]. Available: https://github.com/jianzhangcs/ISTA-Net-PyTorch. 
