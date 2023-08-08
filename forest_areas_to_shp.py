#Imports
import gdal
import ogr
import numpy as np
from scipy import ndimage
from osgeo import ogr, osr

# Helper functions

def filter_green_areas(input_raster, output_raster, moving_window_size=6, neighbor_quote=0.9):
    """
    Check if a TIF has green areas and create a TIF, which contains all green areas.

    Parameter:
    - input_raster: Input TIF to analyse of green areas.
    - output_raster: Name of the output raster.
    - moving_window_size: Size of the window in which cells are considered for classification.
    - neighbor_quote: Percentage of cells that belong to the class in the moving window in order to match unassociated cells.
    """
    # Open the raster
    raster = gdal.Open(input_raster)
    if raster is None:
        print('Could not open ' + input_raster)
        return
    if not 0 <= neighbor_quote <= 1:
        print(f"neighbor_quote: {neighbor_quote} is not in the Interval between 0-1")
        return
    # Read the bands
    red_band = raster.GetRasterBand(1).ReadAsArray()
    green_band = raster.GetRasterBand(2).ReadAsArray()
    blue_band = raster.GetRasterBand(3).ReadAsArray()

    # Find green pixels
    green_pixels = np.logical_and(green_band > red_band, green_band > blue_band)

    # Create a structuring element (a nxn window)
    structure = np.ones((moving_window_size, moving_window_size))

    # Convolve the image with the structure to get the average of the neighborhood
    green_areas = ndimage.binary_opening(green_pixels, structure=structure)

    # Create a new raster with the same properties as the original
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_raster, raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_Byte)

    # Set the geographical information
    out_raster.SetGeoTransform(raster.GetGeoTransform())
    out_raster.SetProjection(raster.GetProjection())

    # Change Pixel in the green areas to green if they are surrounded by the set amount (neighbor_quote) green pixels
    green_neighborhoods = ndimage.convolve(green_areas.astype(float), structure, mode='constant', cval=0.0)
    green_areas = (green_neighborhoods > neighbor_quote)

    # Write the green band into the new raster
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(green_areas)

    # Close the raster
    raster = None
    out_raster = None


def raster_to_vector(raster_path, vector_path):
    """
    Create a vector file outof a raster file

    Parameters:
    - raster_path: Path to raster image for vectorize.
    - vector_path: Path to the created vector file.
    """
    # Open Raster
    raster = gdal.Open(raster_path)

    # Get projection
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjection())

    # Use the first band
    band = raster.GetRasterBand(1)

    # Create a new vector layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = driver.CreateDataSource(vector_path)
    dst_layer = dst_ds.CreateLayer('', srs=srs)

    # vectorize
    field_def = ogr.FieldDefn("RasterVal", ogr.OFTInteger)
    dst_layer.CreateField(field_def)
    gdal.Polygonize(band, None, dst_layer, 0, [], callback=None)

    # Close the data scources
    raster = None
    dst_ds = None


def export_polygons(input_vector_path, output_vector_path, raster_value):
    # Open input layer
    in_driver = ogr.GetDriverByName('ESRI Shapefile')
    in_dataSource = in_driver.Open(input_vector_path, 0)  # 0 bedeutet schreibgeschützt
    in_layer = in_dataSource.GetLayer()
    srs = in_layer.GetSpatialRef()

    # create output layer
    out_driver = ogr.GetDriverByName('ESRI Shapefile')
    if out_driver.DeleteDataSource(output_vector_path) != 0:
        print('Failed to delete existing output file')
    out_dataSource = out_driver.CreateDataSource(output_vector_path)
    out_layer = out_dataSource.CreateLayer('', srs=srs, geom_type=ogr.wkbPolygon)
    # Append attribute "RasterVal"

    in_layer_def = in_layer.GetLayerDefn()
    for i in range(0, in_layer_def.GetFieldCount()):
        out_layer.CreateField(in_layer_def.GetFieldDefn(i))

    # Filter the attributes
    in_layer.SetAttributeFilter(f"RasterVal = {raster_value}")

    # Copy the selected features in a new shape file
    for in_feature in in_layer:
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        out_feature.SetGeometry(in_feature.GetGeometryRef())
        for i in range(0, in_layer_def.GetFieldCount()):
            out_feature.SetField(in_layer_def.GetFieldDefn(i).GetNameRef(), in_feature.GetField(i))
        out_layer.CreateFeature(out_feature)
        out_feature = None

    # close the data sources
    in_dataSource = None
    out_dataSource = None


def merge_all_rasters(input_raster_list, output_raster):
    """Merge raster images with GDAL.

    Parameters:
    - input_raster_list: list of raster images.
    - output_raster: path to result
    """

    # Mosaik Raster mit GDAL
    vrt = gdal.BuildVRT('/vsimem/temp.vrt', input_raster_list, srcNodata=0)
    gdal.Translate(output_raster, vrt)
    vrt = None  # Freigabe der Ressource



# Final function

def get_forest_areas_form_vogteimaps(input_rasters, moving_window_size=6, neighbor_quote=0.9):
    """
    Create a Shape File outof Green Areas in a TIF.

    Paramter:
    - input_rasters: List of input rasters
    - moving_window_size: Size of the window in which cells are considered for classification.
    - neighbor_quote: Percentage of cells that belong to the class in the moving window in order to match unassociated cells.

    Output:
    - create a result_forest_areas.shp

    """
    if len(input_rasters) > 1:

        try:
            merge_all_rasters(input_rasters, "merged_raster.tif")
            filter_green_areas('merged_raster.tif', 'merged_forest_areas.tif', moving_window_size, neighbor_quote)
            raster_to_vector('merged_forest_areas.tif', 'merged_forest_areas.shp')
            export_polygons('merged_forest_areas.shp', 'result_forest_areas.shp', 1)
            return f"Shapefile mit allen Waldstandorten mit den Parametern: moving_window_size = {moving_window_size}, neighbor_quote= {neighbor_quote} wurde erstellt"
        except Exception as E:
            return "Etwas bei der Erstellung des Layers hat nicht funktioniert, stellen Sie sicher, dass alle verwendeten Dateien geschlossen sind", E


    else:
        try:
            filter_green_areas(input_rasters[0], 'forest_areas.tif', moving_window_size, neighbor_quote)
            raster_to_vector('forest_areas.tif', 'forest_areas.shp')
            export_polygons('forest_areas.shp', 'result_forest_areas.shp', 1)
            return f"Shapefile mit allen Waldstandorten mit den Parametern: moving_window_size = {moving_window_size}, neighbor_quote= {neighbor_quote} wurde erstellt"
        except Exception as E:
            return "Etwas bei der Erstellung des Layers hat nicht funktioniert, stellen Sie sicher, dass alle verwendeten Dateien geschlossen sind", E


if __name__ == "__main__":
    #Paths to the Tif Data
    input_raster_list = ['../2814_Zwischenahn_modifiziert.tif', '../2713_Westerstede_modifiziert.tif',
                         '../2714_Wiefelstede_modifiziert.tif', '../2715_Rastede_modifiziert.tif',
                         '../2813_Edewecht_modifiziert.tif', '../2815_Oldenburg_modifiziert.tif']
    get_forest_areas_form_vogteimaps(input_raster_list, 10, 0.7)