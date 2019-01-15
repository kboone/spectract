# spectract
Forward modeling code for IFU spectrographs

This code implements a forward model for IFU spectrographs. It is highly experimental, and is mostly limited to the SNIFS instrument for now. I do not recommend trying to use this package on other data.

The code builds an optical model of the spectrograph from a series of engineering data. This includes modeling the profiles of the wings of each spaxel that are often ignored but that can leak into adjacent spaxels.

For new images, the code tweaks the model using an arc image taken before or after the science image, and then does a limited tweak on the actual science image before extracting the data cube.
