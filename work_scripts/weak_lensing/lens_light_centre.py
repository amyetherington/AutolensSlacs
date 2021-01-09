import autolens as al
import autolens.plot as aplt
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
"""
This tool allows one to input the lens light centre(s) of a strong lens(es), which can be used as a fixed value in
pipelines. 

First, we set up the dataset we want to mark the lens light centre of.
"""


dataset_name = "2341+0000"

"""
The path where the dataset will be loaded from, which in this case is:

 `/autolens_workspace/dataset/imaging/with_lens_light/light_sersic__mass_sie__source_sersic`
"""

dataset_path = "/Users/dgmt59/Documents/Data/slacs_weak_lensing/"

output_path = "/Users/dgmt59/PycharmProjects/autolens_slacs_pre_v_1/dataset/slacs_shu/slacs"

"""If you use this tool for your own dataset, you *must* double check this pixel scale is correct!"""

pixel_scales = 0.03

"""
When you click on a pixel to mark a position, the search box looks around this click and finds the pixel with
the highest flux to mark the position.

The `search_box_size` is the number of pixels around your click this search takes place.
"""

search_box_size = 5

imaging = al.Array.from_fits(
    file_path=f"{dataset_path}{dataset_name}_F814W_drz_sci.fits",
    pixel_scales=pixel_scales,
)
image_2d = imaging.in_2d


"""
This code is a bit messy, but sets the image up as a matplotlib figure which one can double click on to mark the
positions on an image.
"""

light_centres = []


def onclick(event):
    if event.dblclick:
        print(event.ydata, event.xdata)

        y_pixels = int(event.ydata)
        x_pixels = int(event.xdata)

        flux = -np.inf

        for y in range(y_pixels - search_box_size, y_pixels + search_box_size):
            for x in range(x_pixels - search_box_size, x_pixels + search_box_size):
                flux_new = image_2d[y, x]
                #      print(y, x, flux_new)
                if flux_new > flux:
                    flux = flux_new
                    y_pixels_max = y
                    x_pixels_max = x

        grid_arcsec = image_2d.geometry.grid_scaled_from_grid_pixels_1d(
            grid_pixels_1d=al.Grid.manual_2d(
                grid=[[[y_pixels_max + 0.5, x_pixels_max + 0.5]]],
                pixel_scales=pixel_scales,
            )
        )
        y_arcsec = grid_arcsec[0, 0]
        x_arcsec = grid_arcsec[0, 1]

        print("clicked on:", y_pixels, x_pixels)
        print("Max flux pixel:", y_pixels_max, x_pixels_max)
        print("Arc-sec Coordinate", y_arcsec, x_arcsec)

        light_centres.append((y_pixels_max, x_pixels_max))


n_y, n_x = imaging.shape_2d
hw = int(n_x / 2) * pixel_scales
ext = [-hw, hw, -hw, hw]
fig = plt.figure(figsize=(14, 14))
plt.imshow(imaging.in_2d, cmap='gray', norm=SymLogNorm(vmin=-0.13, vmax=20, linthresh=0.02))
plt.xlim(3500, 4000)
plt.ylim(5000, 5500)
plt.colorbar()
cid = fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)
plt.close(fig)

light_centres = al.GridIrregularGrouped(grid=light_centres)

print(light_centres)

"""
Now lets plot the image and lens light centre, so we can check that the centre overlaps the brightest pixel in the
lens light.
"""
#aplt.Array(array=imaging, light_profile_centres=light_centres)

"""
Now we`re happy with the lens light centre(s), lets output them to the dataset folder of the lens, so that we can 
load them from a.dat file in our pipelines!
"""
light_centres.output_to_file(
    file_path=f"{output_path}{dataset_name}/lens_centre_pix.dat", overwrite=True
)
