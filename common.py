import shapefile
import numpy as np
import shapely.geometry

def get_coastline():
    # load coastline
    shape_reader = shapefile.Reader('/home/tom/data/ordnance_survey/strategi/data/coastline')

    coastline_index = [2400, 2403, 2404, 2405, 2408, 2399, 2406, 2407, 2402, 2401]
    coastline = np.array(sum(
        [x.points for i, x in enumerate(shape_reader.shapes()) if i in coastline_index], []
    ))
    coastline = shapely.geometry.Polygon(coastline)
    coastline = shapely.geometry.polygon.orient(coastline)
    assert coastline.is_valid
    return coastline
