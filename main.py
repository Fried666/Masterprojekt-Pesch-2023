import shapely
import matplotlib.pyplot as plt
from geocube.api.core import make_geocube
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import rioxarray
import os
def check_nan(value):
    return value != value

def convert_vector_to_raster(gdf: gpd.GeoDataFrame, template_raster, path_to_images:str,
                             filename:str) -> None:
    """Create Raster based on Geodataframe and Template Raster.

    Shapes will be Vectorized based on the Resolution and the Snippet of the template Raster

    :param gdf: GeoDataFrame containing Vector Shapes (not points)
    :param template_raster: Raster which defines Meta (including coordinate-reference-system, resolution and snippet)
    :param path_to_images: Directory where the image should be saved
    :param filename: filename of the output image
    """

    ## Get features that intersect w raster bounds
    ## reproject the buffered template bounds to the deforestation vector projection
    bounds = template_raster.rio.bounds()
    gpd_bounds = gpd.GeoDataFrame({"id":1,"geometry":[shapely.geometry.box(*bounds)]})
    gpd_bounds_reproj = gpd_bounds.set_crs(template_raster.rio.crs).to_crs(gdf.crs)
    ## now we perform the selection based on intersection
    print("        Select Vectors which intersect with Template-Raster")
    vectors_inbounds = gdf.overlay(gpd_bounds_reproj,how='intersection')

    # reproject the selected vectors to the template raster CRS
    vectors_inbounds_reproj = vectors_inbounds.to_crs(template_raster.rio.crs)
    # Rasterize
    vectors_inbounds_reproj['flag']=1
    vectors_inbounds_reproj = vectors_inbounds_reproj[['flag','geometry']]
    #read the resolution of the template raster
    resolution_size_x = template_raster.rio.transform()[0]
    resolution_size_y = template_raster.rio.transform()[4]
    # Now we rasterize
    print("        Convert Vectors to Raster")
    rasterized_vectors = make_geocube(
        vector_data=vectors_inbounds_reproj,
        measurements=["flag"],
        resolution=(resolution_size_y,resolution_size_x),
        output_crs = template_raster.rio.crs,
        geom = gpd_bounds_reproj.iloc[[0]].to_json()
    )
    print("        Save to file")
    rasterized_vectors.rio.to_raster(path_to_images+filename)

    return()

def create_raster_from_shape(resolution: int, gdf: gpd.GeoDataFrame, filename: str) -> None:
    """Create Raster based on Geodataframe containing vector information.

    All Shapes will be vecorized with the "burning value" 1.

    :param resolution: Specifies Resolution in meter
    :param gdf: GeoDataFrame containing Vector Shapes (not points)
    :param path_to_images: Directory where the image should be saved
    :param filename: filename of the output image
    :return:
    """
    # Read California Census Polygon Shapefile
    gdf["burn_value"]=1
    gdf = gdf.to_crs('EPSG:25832')

    print("Rasterizing...")
    # Using GeoCube to rasterize the Vector
    generated_raster = make_geocube(
        vector_data=gdf,
        measurements=["burn_value"],
        resolution=(resolution,resolution),
        fill = 0
    )

    print("Saving Raster...")
    # Save raster census raster
    generated_raster.rio.to_raster(filename)
    print("finished")

def display_bands(tif_path:str) -> None:
    # Open with rasterio
    with rasterio.open(tif_path) as src:
        # Get number of bands
        num_bands = src.count

        # Iterate bands
        for band_num in range(1, num_bands + 1):
            # Load band
            raster_array = src.read(band_num)

            # Show band
            plt.imshow(raster_array, cmap='gray')
            plt.colorbar()
            plt.title(f"Band {band_num}")
            plt.show()

def save_numpy_array_as_geotiff_by_reference_raster(array: np.ndarray, reference_raster_path: str, output_path: str, nodata_value:int= None) -> None:
    with rasterio.open(reference_raster_path) as src:
        # Get georeferenced metadata from the reference raster
        meta = src.meta.copy()

        # Replace the data array with the new NumPy array
        meta['dtype'] = array.dtype
        meta['count'] = 1
        meta['height'], meta['width'] = array.shape

        # Optionally set nodata value
        if nodata_value is not None:
            meta['nodata'] = nodata_value

        # Save the array as GeoTIFF
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(array, 1)

class GeoDataProcessing:
    def __init__(self,
                 path_to_excel: str = "",
                 path_to_geodatabase: str = "",
                 template_raster_name: str = "",
                 animal_name: str = "",
                 resolution_m: int = 5,
                 project_folder_path=""):

        # Initialize Information for process
        template_raster_path = project_folder_path + template_raster_name
        path_to_images = project_folder_path + animal_name + f"\\resolution_{resolution_m}m" + "\\"
        shapefiles = pd.read_excel(path_to_excel, sheet_name=animal_name)

        # Check if folders already exist -> if not create
        if not os.path.exists(project_folder_path): os.makedirs(project_folder_path)
        if not os.path.exists(path_to_images): os.makedirs(path_to_images)

        # Check if template_raster already exist -> if not create
        if not os.path.exists(template_raster_path):
            create_raster_from_shape(resolution=resolution_m,
                                          gdf=gpd.read_file(path_to_geodatabase, layer="LK_Ammerland"),
                                          filename=template_raster_path)

        template_raster = rioxarray.open_rasterio(template_raster_path)

        # Generate Cost Raster for each Vectorfile (based on Input Excel) - store values in class and external
        self.layers = {}
        for index, row in shapefiles.iterrows():
            self.layers[row["name"]] = self.InitializeLayer(row, path_to_geodatabase, path_to_images, template_raster)

        # Iterate every raster and calculating the average cost value per cell
        np.seterr(invalid='ignore')
        my_score = np.empty(list(template_raster.shape[1:]))  # Create empty array with the dimension of template raster
        my_amount = np.empty(
            list(template_raster.shape[1:]))  # Create empty array with the dimension of template raster
        for picture in self.layers:
            self.layers[picture].rasterized_image_array[np.isnan(self.layers[picture].rasterized_image_array)] = 0
            my_score = my_score + (self.layers[picture].rasterized_image_array * self.layers[picture].cost_value)
            my_amount = my_amount + self.layers[picture].rasterized_image_array

        # Save the amount for test reasons
        self.amount_of_overlaying_shapes = my_amount
        save_numpy_array_as_geotiff_by_reference_raster(my_amount, template_raster_path,
                                                             project_folder_path + f"Amount_of_overlaying_Shapes_{resolution_m}m_{animal_name}_with_forests.tif")
        result_raster_array = np.divide(my_score, my_amount, out=np.zeros_like(my_score))

        # Get old forests as vector and raster (based on study area)
        old_forests_vector = gpd.read_file(path_to_geodatabase, layer="Alte_Waldstandorte")
        temp_filename = "old_forests.tif"
        convert_vector_to_raster(gdf=old_forests_vector, template_raster=template_raster,
                                      path_to_images=path_to_images, filename=temp_filename)
        old_forests_layer_raw = rasterio.open(path_to_images + temp_filename)
        old_forests_layer_array = old_forests_layer_raw.read(1)
        old_forests_layer_raw.close()
        old_forests_layer_array[np.isnan(old_forests_layer_array)] = 0

        # Cut raster by study area
        template_raster_raw = rasterio.open(template_raster_path)
        template_raster_array = template_raster_raw.read(1)
        template_raster_raw.close()
        template_raster_array[np.isnan(template_raster_array)] = 0
        result_raster_array[np.isnan(result_raster_array)] = 0
        result_cutted_to_size = template_raster_array * result_raster_array
        # set None where no values available:
        result_cutted_to_size[result_cutted_to_size == 0] = None
        self.result_raster_array_with_forests_forests_0_missing_value_None = result_cutted_to_size
        # set 0 where old forests:
        result_cutted_to_size[old_forests_layer_array == 1] = 0.001
        save_numpy_array_as_geotiff_by_reference_raster(result_cutted_to_size, template_raster_path,
                                                             project_folder_path + f"result_cost_raster_{resolution_m}m_{animal_name}_with_forests.tif")

        self.result_raster_array_with_forests = result_cutted_to_size

        # Generate result intersecting with historical forest
        filename = "historical_forests" + ".tif"
        hist_gdf = gpd.read_file(path_to_geodatabase, layer="vogtei_waldstandorte")
        # smaller errors accured by rasterizing: workaround
        hist_gdf = hist_gdf[hist_gdf["geometry"] != None]
        convert_vector_to_raster(gdf=hist_gdf, template_raster=template_raster, path_to_images=path_to_images,
                                      filename=filename)
        historical_forests_layer_raw = rasterio.open(path_to_images + filename)
        historical_forests_layer_array = historical_forests_layer_raw.read(1)
        historical_forests_layer_raw.close()
        historical_forests_layer_array[np.isnan(historical_forests_layer_array)] = 0
        # set 0one where No historical forest or actual forest
        result_cutted_to_size[(historical_forests_layer_array == 0) & (old_forests_layer_array == 0)] = None
        save_numpy_array_as_geotiff_by_reference_raster(result_cutted_to_size, template_raster_path,
                                                             project_folder_path + f"result_cost_raster_{resolution_m}m_{animal_name}_with_forests_cutted_by_historical_forests.tif")
        self.result_raster_array_with_forests_cutted_hist = result_cutted_to_size

    class InitializeLayer:
        def __init__(self, row, path_to_geodatabase, path_to_images, template_raster):
            """Rasterize Layer based on Template and initialize meta Information"""

            # Initialize general Information
            self.name = row["name"]
            print(f"Initialize Shape: {self.name}:")
            self.cost_value = row["cost_value"]
            self.feature_dataset = row["feature_dataset"]
            self.feature_class = row["feature_class"]
            self.attribute = row["attribute"]
            self.attribute_value = row["attribute_value"]
            self.buffer = row["buffer"]
            self.comment = row["comment"]
            filename = self.name + ".tif"

            if not os.path.exists(path_to_images + filename):

                print("    Reading from Geodatabase", end=" - ")
                gdf = gpd.read_file(path_to_geodatabase, layer=row["feature_class"])
                print(f"finished - number of elements: {len(gdf)}")

                self.raw_gdf = gdf
                if not check_nan(row["attribute"]):
                    print(f"""    Filter by Attribute: {row["attribute"]} == {row["attribute_value"]} """, end=" - ")
                    # some rows continue str some continue floats:
                    if type(gdf[row["attribute"]].iloc[0]) == np.float64:
                        gdf_after_filter = gdf[gdf[row["attribute"]] == row["attribute_value"]]
                    else:
                        gdf_after_filter = gdf[gdf[row["attribute"]] == str(int(row["attribute_value"]))]
                    self.raw_gdf = gdf_after_filter
                    print(f"finished - number of elements: {len(gdf_after_filter)}")
                else:
                    print("    No filter required")

                # Buffer (if needed)
                self.buffered_gdf = self.raw_gdf
                if not check_nan(row["buffer"]):
                    print(f"""    Calculate Buffer with distance of {row["buffer"]}m""", end=" - ")
                    gdf_after_filter['geometry'] = gdf_after_filter['geometry'].buffer(row["buffer"])
                    self.buffered_gdf = self.raw_gdf
                    print("finished")
                else:
                    print("    No buffer required")

                # Rasterize
                print("    Calculate Raster:")
                convert_vector_to_raster(gdf=self.buffered_gdf, template_raster=template_raster,
                                              path_to_images=path_to_images, filename=filename)

            raster = rasterio.open(path_to_images + filename)
            # everything is saved on band1 - uncomment if u want access (storage required)
            self.rasterized_image_array = raster.read(1)
            raster.close()
            print("finished")
            print()
