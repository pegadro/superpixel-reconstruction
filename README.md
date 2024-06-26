# Superpixel Reconstruction
Reconstruct images from the superpixels of a set of images that do not contain the image to be reconstructed.

Images from which superpixels were obtained:
<div style="display: flex; flex-direction: row; align-items: center;">
  <img src="images_test/monalisa/Cesareborgia.jpeg" alt="Cesare Borgia" width="300px" style="margin-right: 10px"/>

  <img src="images_test/monalisa/ermine.jpeg" alt="Ermine" width="300px" />
</div>
Target image:<br>
<img src="images_test/monalisa/monalisa.jpeg" alt="Monalisa" width="300px" />
<br>Reconstrucion:<br>
<img src="reconstruction/monalisa_reconstruction.png" alt="Monalisa" width="300px" />

## Summary

The project involves reconstructing images from the superpixels of a set of images that do not contain the image being recreated. The SLIC (Simple Linear Iterative Clustering) algorithm is used to obtain the superpixels of the images. The reconstruction is carried out using Simulated Annealing, with the aim of minimizing the mean squared error (or maximizing the structural similarity index measure) between the target image and the reconstructed image.

## How it works

To know about how it works, the methodology followed, the methods proposed and implemented, read the report.

[Report](report_en.pdf)

## More examples

<div style="display: flex; flex-direction: column; justify-content: space-around;">
  <img src="images_test/beatles.jpg" alt="The Beatles" width="400px" />
  <br>
  <img src="reconstruction/beatles_reconstruction.png" alt="The Beatles reconstructed" width="400px" />
  <br>
  <span>Reconstructed with the superpixels from the images below</span>
  <br>
  <img src="images_test/pearl.jpeg" alt="The Beatles reconstructed" width="400px" />
  <br>
  <img src="images_test/desperate_man.jpeg" alt="The Beatles reconstructed" width="400px" />
</div>
