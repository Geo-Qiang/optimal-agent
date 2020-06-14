def pad_data(data, nei_size):
    m, n = data.shape
    t1 = np.zeros([nei_size // 2, n])
    data = np.concatenate((t1, data, t1))
    m, n = data.shape
    t2 = np.zeros([m, nei_size // 2])
    data = np.concatenate((t2, data, t2), axis=1)
    return data


def gen_data(data, nei_size):
    x, y = data.shape
    m = x - nei_size // 2 * 2
    n = y - nei_size // 2 * 2
    res = np.zeros([m * n, nei_size ** 2])
    k = 0
    for i in range(nei_size // 2, m + nei_size // 2):
        for j in range(nei_size // 2, n + nei_size // 2):
            res[k, :] = np.reshape(
                data[i - nei_size // 2:i + nei_size // 2 + 1, j - nei_size // 2:j + nei_size // 2 + 1].T, (1, -1))
            k += 1
    return res


def raster_to_polygon(input_path):
    input_file = input_path + "land_use.tif"
    dst_filename = input_path + "land_use.shp"
    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    src_band = ds.GetRasterBand(1)
    mask_band = src_band.GetMaskBand()
    drv = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = drv.CreateDataSource(dst_filename)
    srs = None
    dst_layer_name = 'out'
    dst_layer = dst_ds.CreateLayer(dst_layer_name, srs=srs)
    dst_field_name = 'DN'
    fd = ogr.FieldDefn(dst_field_name, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0
    options = []
    gdal.Polygonize(src_band, mask_band, dst_layer, dst_field, options)


def shp_to_raster(input_path):
    shp_patch = input_path + "land_use.shp"
    shp_patch_t = input_path + "land_use_rural_city.shp"
    land_use_background = input_path + "land_use.tif"
    output = input_path + "land_use_rural_city.tif"
    field = "new"
    nodata = 0
    shp_process(shp_patch, shp_patch_t)
    ndsm = land_use_background
    data = gdal.Open(ndsm, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    proj = data.GetProjection()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    mb_v = ogr.Open(shp_patch_t)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]
    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Int16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    no_data_value = nodata
    band.SetNoDataValue(no_data_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=%s" % field, 'ALL_TOUCHED=TRUE'])
    target_ds = None
    return output


def shp_process(shp_patch_old, shp_patch_new):
    gdf = geopandas.read_file(shp_patch_old)
    urban = land_use_code["urban"]
    rural = land_use_code["rural"]
    arable = land_use_code["arable"]
    rural_c = 0
    city_c = 10000
    arable_c = 20000
    gdf["new"] = 0
    for i in range(0, len(gdf)):
        geo = gdf.DN[i]
        if geo == urban:
            city_c += 1
            gdf.loc[i, 'new'] = city_c
        elif geo == rural:
            rural_c += 1
            gdf.loc[i, 'new'] = rural_c
        elif geo == arable:
            rural_c += 1
            gdf.loc[i, 'new'] = arable_c
        else:
            gdf.loc[i, 'new'] = 0
    gdf.to_file(shp_patch_new)


def normalization(land_use_this, land_use_edge, a, b, ratio):
    x = economy(land_use_this, a, b)
    y = ecology(land_use_this, a, b)
    z = compact(land_use_edge, land_use_edge, a, b)
    x_test = minmax_scale(x, feature_range=[0, 1], axis=0, copy=True)
    y_test = minmax_scale(y, feature_range=[0, 1], axis=0, copy=True)
    z_test = minmax_scale(z, feature_range=[0, 1], axis=0, copy=True)
    res = ratio[0] * x_test + ratio[1] * y_test + ratio[2] * z_test
    return res


def roulette_selection(fitness):
    sum_fits = sum(fitness)
    rnd_point = random.uniform(0, sum_fits)
    accumulator = 0.0
    for ind, val in enumerate(fitness):
        accumulator += val
        if accumulator >= rnd_point:
            return ind


def optimization_de_agent(region_data_package, parameter_package):
    land_use_this = region_data_package["land_use"]
    land_use_edge = region_data_package["land_use_edge"]
    arable = land_use_code["arable"]
    rural = land_use_code["rural"]
    ratio = parameter_package["ratio"]
    optimal_len = parameter_package["optimal_len"]
    arable_fitness = region_data_package["arable"]
    city_fitness = region_data_package["city"]
    rural_fitness = region_data_package["rural"]
    raster_plate = region_data_package["raster_plate"]
    np_plate = np.unique(raster_plate)
    f_worst_list = []
    for i_plate in np_plate:
        a = np.where(raster_plate == i_plate)[0]
        b = np.where(raster_plate == i_plate)[1]
        np_size = np.size(a)
        influence_plate = 0
        if land_use_this[a[0], b[0]] == arable:
            for i in range(np_size):
                temp_value = arable_fitness[a[i], b[i]]
                influence_plate += temp_value
        elif land_use_this[a[0], b[0]] == rural:
            for i in range(np_size):
                temp_value = rural_fitness[a[i], b[i]]
                influence_plate += temp_value
        f_worst = influence_plate / np_size + normalization(land_use_this, land_use_edge, a, b, ratio)
        f_worst_list.append(f_worst)
    index_low = np_plate[(np.argsort(f_worst_list))[0]]
    index_low_new = index_low[0:optimal_len]
    m, n = land_use_this.shape
    land_use_index = np.zeros([m, n])
    for item in index_low_new:
        low_a = np.where(raster_plate == np_plate[index_low[item]])[0]
        low_b = np.where(raster_plate == np_plate[index_low[item]])[1]
        land_use_index[low_a, low_b] = 1
    return land_use_index


def construction_de_agent(region_data_package):
    city_edge_urban = 4
    city_edge_rural = 3
    land_use_this = region_data_package["land_use"]
    urban = land_use_code["urban"]
    rural = land_use_code["rural"]
    land_use_edge_this = region_data_package["land_use_edge "]
    suitableness_urban = region_data_package["urban"]
    suitableness_rural = region_data_package["rural"]
    m, n = land_use_this.shape
    index_urban = np.zeros([m, n])
    index_rural = np.zeros([m, n])
    for item in range(m * n):
        x = item // n
        y = item % n
        roulette = np.random.rand()
        if np.sum(land_use_edge_this[item] == urban) >= city_edge_urban and land_use_this[x, y] != urban:
            if roulette < suitableness_urban[x, y]:
                index_urban[x, y] = 1
        elif np.sum(land_use_edge_this[item] == rural) >= city_edge_rural and land_use_this[x, y] != rural:
            if roulette < suitableness_rural[x, y]:
                index_rural[x, y] = 1
    land_use_plan = {
        "index_urban": index_urban,
        "index_rural": index_rural,
    }
    return land_use_plan


def ecology_de_agent(region_data_package):
    ecology_edge_in = 2
    land_use_this = region_data_package["land_use"]
    forest = land_use_code["forest"]
    grassland = land_use_code["grassland"]
    orchard = land_use_code["orchard"]
    m, n = land_use_this.shape
    land_use_edge_this = region_data_package["land_use_edge "]
    suitableness_forest = region_data_package["forest"]
    suitableness_grassland = region_data_package["grassland"]
    suitableness_orchard = region_data_package["orchard"]
    index_forest = np.zeros([m, n])
    index_grassland = np.zeros([m, n])
    index_orchard = np.zeros([m, n])
    for item in range(m * n):
        x = item // n
        y = item % n
        roulette = np.random.rand()
        if np.sum(land_use_edge_this[item] == forest) >= ecology_edge_in and land_use_this[x, y] != forest:
            if roulette < suitableness_forest[x, y]:
                index_forest[x, y] = 1
        elif np.sum(land_use_edge_this[item] == grassland) >= ecology_edge_in and land_use_this[x, y] != grassland:
            if roulette < suitableness_grassland[x, y]:
                index_grassland[x, y] = 1
        elif np.sum(land_use_edge_this[item] == orchard) >= ecology_edge_in and land_use_this[x, y] != orchard:
            if roulette < suitableness_orchard[x, y]:
                index_orchard[x, y] = 1
    land_use_plan = {
        "index_forest": index_forest,
        "index_grassland": index_grassland,
        "index_orchard": index_orchard,
    }
    return land_use_plan


def government_decision(region_data_package, parameter_package, land_use_plan_ec, land_use_plan_co, land_use_plan_op):
    index_forest = land_use_plan_ec["index_forest"]
    index_grassland = land_use_plan_ec["index_grassland"]
    index_orchard = land_use_plan_ec["index_orchard"]
    index_urban = land_use_plan_co["index_urban"]
    index_rural = land_use_plan_co["index_rural"]
    land_use_index = land_use_plan_op
    land_use_this = region_data_package["land_use"]
    land_use_plan = land_use_this
    urban = land_use_code["urban"]
    rural = land_use_code["rural"]
    forest = land_use_code["forest"]
    grassland = land_use_code["grassland"]
    orchard = land_use_code["orchard"]
    suitableness_forest = region_data_package["forest"]
    suitableness_grassland = region_data_package["grassland"]
    suitableness_orchard = region_data_package["orchard"]
    suitableness_urban = region_data_package["urban"]
    suitableness_rural = region_data_package["rural"]
    function = region_data_package["function"]
    land_use_plan_comprehensive = index_forest + index_grassland + index_orchard + index_urban + index_rural + land_use_index
    row_index = np.where(land_use_plan_comprehensive > 0)[0]
    col_index = np.where(land_use_plan_comprehensive > 0)[1]
    number = row_index.size
    for item in range(number):
        row_this = row_index[item]
        col_this = col_index[item]
        land_use_name = list(land_use_code.keys())[
            list(land_use_code.values()).index(str(land_use_this[row_this, col_this]))]
        convert_forest = 1 - convert.loc[land_use_name, forest]
        change_forest = index_forest[row_this, col_this] * suitableness_forest[row_this, col_this] * function[
            row_this, col_this] * convert_forest
        convert_grassland = 1 - convert.loc[land_use_name, grassland]
        change_grassland = index_grassland[row_this, col_this] * suitableness_grassland[row_this, col_this] * function[
            row_this, col_this] * convert_grassland
        convert_orchard = 1 - convert.loc[land_use_name, orchard]
        change_orchard = index_orchard[row_this, col_this] * suitableness_orchard[row_this, col_this] * function[
            row_this, col_this] * convert_orchard
        convert_rural = 1 - convert.loc[land_use_name, rural]
        change_rural = index_rural[row_this, col_this] * suitableness_rural[row_this, col_this] * function[
            row_this, col_this] * convert_rural
        convert_urban = 1 - convert.loc[land_use_name, urban]
        change_urban = index_urban[row_this, col_this] * suitableness_urban[row_this, col_this] * function[
            row_this, col_this] * convert_urban
        choose_result = roulette_selection(
            [change_forest, change_grassland, change_orchard, change_rural, change_urban])
        if choose_result == 0:
            if global_limit_package["forest"] > 0:
                land_use_plan[row_this, col_this] = forest
                global_limit_package["forest"] = global_limit_package["forest"] - 1
        elif choose_result == 1:
            if global_limit_package["grassland"] > 0:
                land_use_plan[row_this, col_this] = grassland
                global_limit_package["grassland"] = global_limit_package["grassland"] - 1
        elif choose_result == 2:
            if global_limit_package["orchard"] > 0:
                land_use_plan[row_this, col_this] = orchard
                global_limit_package["orchard"] = global_limit_package["orchard"] - 1
        elif choose_result == 3:
            if global_limit_package["rural"] > 0:
                land_use_plan[row_this, col_this] = rural
                global_limit_package["rural"] = global_limit_package["rural"] - 1
        elif choose_result == 4:
            if global_limit_package["urban"] > 0:
                land_use_plan[row_this, col_this] = urban
                global_limit_package["urban"] = global_limit_package["urban"] - 1
    return land_use_plan


def separate_package(factors_package, main_package, parameter_package, influence_list_package):
    land_use = main_package['land_use']
    region = main_package['region']
    function = main_package['function']
    region_number = parameter_package['number_region']
    develop_region = parameter_package['develop_region']
    convert_index = parameter_package['convert_index']
    raster_factors_agent = cal_factors(factors_package, influence_list_package)
    region_data_package = {}
    path_main_data = parameter_package["Path_main_data"]
    raster_to_polygon(path_main_data)
    raster_plate_patch = shp_to_raster(path_main_data)
    raster_plate = (rasterio.open(raster_plate_patch)).read(1)
    for i in range(0, region_number):
        a = np.where(region == i)[0]
        b = np.where(region == i)[1]
        land_use_region = land_use[a.min():a.max() + 1, b.min():b.max() + 1]
        function_region = function[a.min():a.max() + 1, b.min():b.max() + 1]
        region_region = region[a.min():a.max() + 1, b.min():b.max() + 1]
        city_agent_t = (raster_factors_agent["city"])[a.min():a.max() + 1, b.min():b.max() + 1]
        rural_agent_t = (raster_factors_agent["rural"])[a.min():a.max() + 1, b.min():b.max() + 1]
        arable_agent_t = (raster_factors_agent["arable"])[a.min():a.max() + 1, b.min():b.max() + 1]
        forest_agent_t = (raster_factors_agent["forest"])[a.min():a.max() + 1, b.min():b.max() + 1]
        grassland_agent_t = (raster_factors_agent["grassland"])[a.min():a.max() + 1, b.min():b.max() + 1]
        orchard_agent_t = (raster_factors_agent["orchard"])[a.min():a.max() + 1, b.min():b.max() + 1]
        land_use_edge = pad_data(land_use_region, 3)
        land_use_edge_by = gen_data(land_use_edge, 3)
        raster_plate_t = raster_plate[a.min():a.max() + 1, b.min():b.max() + 1]
        region_data = {
            "land_use": land_use_region,
            "function": function_region,
            "region": region_region,
            'city': city_agent_t,
            'rural': rural_agent_t,
            'arable': arable_agent_t,
            'forest': forest_agent_t,
            'grassland': grassland_agent_t,
            'orchard': orchard_agent_t,
            "land_use_edge ": land_use_edge_by,
            "develop_region": develop_region,
            "convert_index": convert_index,
            "raster_plate": raster_plate_t
        }
        region_data_package[i] = region_data
    return region_data_package


def group_package(main_package, parameter_package, region_data_package):
    region_number = parameter_package['number_region']
    region = main_package['region']
    land_use = main_package['land_use']
    m, n = land_use.shape()
    result = np.zeros((m, n))
    for i in range(0, region_number):
        a = np.where(region == i)[0]
        b = np.where(region == i)[1]
        result[a.min():a.max() + 1, b.min():b.max() + 1] = region_data_package[i]['land_use']
    return result


def government_agent(factors_package, main_package, parameter_package, limit_package, influence_list_package,
                     land_use_code_package):
    global global_limit_package
    global_limit_package = limit_package
    global land_use_code
    land_use_code = land_use_code_package
    region_data_package = separate_package(factors_package, main_package, parameter_package, influence_list_package)
    year = parameter_package['number_year']
    region_number = parameter_package['number_region']
    for item_y in range(year):
        for item_r in range(0, region_number):
            rural_pri = len(np.where(region_data_package[item_r]["land_use"] == land_use_code_package["rural"])[0])
            city_pri = len(np.where(region_data_package[item_r]["land_use"] == land_use_code_package["urban"])[0])
            land_use_plan_ec = ecology_de_agent(region_data_package[item_r])
            land_use_plan_co = construction_de_agent(region_data_package[item_r])
            land_use_plan_op = optimization_de_agent(region_data_package[item_r], parameter_package)
            land_use_decision = government_decision(region_data_package[item_r], parameter_package, land_use_plan_ec,
                                                    land_use_plan_co, land_use_plan_op)
            rural_boost = (len(np.where(land_use_decision == land_use_code_package["rural"])[0]) - rural_pri)
            city_boost = (len(np.where(land_use_decision == land_use_code_package["urban"])[0]) - city_pri)
            region_data_package[item_r]["land_use"] = land_use_decision
            print("农村新增:" + str(rural_boost), "城市新增:" + str(city_boost))
        result_y = group_package(main_package, parameter_package, region_data_package)
        cc = str(item_y)
        profile_agent = parameter_list["raster_profile"]
        profile_agent.update(dtype=result_y.dtype, count=1)
        with rasterio.open(parameter_list["Path_result"] + cc + '.tif', mode='w', **profile) as dst:
            dst.write(result_y, 1)


def open_raster(folder_path):
    file_list = os.listdir(folder_path)
    inf_list = {}
    for inf in file_list:
        inf_name = inf.split('.')[0]
        inf_list[inf_name] = (rasterio.open(folder_path + inf)).read(1)
        print(inf)
    return inf_list


def cal_factors(raster_data, factor_list):
    land_use = raster_data["land_use"]
    m, n = land_use.shape
    factor_in_group = {}
    for k in factor_list.keys():
        current_factor_list = factor_list[k]
        temp = np.zeros((m, n))
        for idx, item in enumerate(raster_data.values()):
            temp = temp + (item * current_factor_list[idx])
        factor_in_group[k] = temp
    return factor_in_group


def input_excel(patch_n):
    name = pd.read_excel(patch_n)
    return name


def economy(land_use_this, a, b):
    land_use_economy = {
        'traffic': 600.2,
        'urban': 5193.5,
        'water': 47.088,
        'unused': 9,
        'rural': 4000,
        'indus': 6000,
        'arable': 529.089,
        'grassland': 64.0989,
        'forest': 44.028,
        'orchard': 529.089
    }
    if a == 0 and b == 0:
        fitness = 0
        land_use_type = land_use_economy.keys()
        for i in land_use_type:
            land_use_count = np.count_nonzero(land_use_this == land_use_code[i])
            fitness += land_use_count * land_use_economy[i]
        return fitness
    else:
        np_size = np.size(a)
        if land_use_this[a[0], b[0]] == land_use_code["arable"]:
            fitness = np_size * land_use_economy["arable"]
            return fitness
        elif land_use_this[a[0], b[0]] == land_use_code["city"]:
            fitness = np_size * land_use_economy["city"]
            return fitness
        elif land_use_this[a[0], b[0]] == land_use_code["rural"]:
            fitness = np_size * land_use_economy["rural"]
            return fitness


def ecology(land_use_this, a, b):
    land_use_ecology = {
        'traffic': 10,
        'urban': 93.39,
        'water': 3660.894,
        'unused': 40.446,
        'rural': 33.39,
        'indus': 10,
        'arable': 665.874,
        'grassland': 697.61,
        'forest': 8105.478,
        'orchard': 1740.06
    }
    if a == 0 and b == 0:
        fitness = 0
        land_use_type = land_use_ecology.keys()
        for i in land_use_type:
            land_use_count = np.count_nonzero(land_use_this == land_use_code[i])
            fitness += land_use_count * land_use_ecology[i]
        return fitness
    else:
        np_size = np.size(a)
        if land_use_this[a[0], b[0]] == land_use_code["arable"]:
            fitness = np_size * land_use_ecology["arable"]
            return fitness
        elif land_use_this[a[0], b[0]] == land_use_code["city"]:
            fitness = np_size * land_use_ecology["city"]
            return fitness
        elif land_use_this[a[0], b[0]] == land_use_code["rural"]:
            fitness = np_size * land_use_ecology["rural"]
            return fitness


def compact(land_use_this, land_use_edge, a, b):
    this_type = land_use_this[a[0], b[0]]
    m, n = land_use_this.shape
    edge_value = 0
    for item in range(m * n):
        if m in a and n in b:
            edge_value += np.sum(land_use_edge[item] == this_type)
    return edge_value


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import random
    from sklearn.preprocessing import minmax_scale
    import rasterio
    import time
    import os
    import gdal
    from osgeo import ogr
    import gdalconst
    import geopandas

    Folder_Path_main = 'D:/MASA/main_raster/'
    Folder_Path_factor = 'D:/MASA/influence/'
    Folder_Path_develop_room = 'D:/MASA/develop_room.excel'
    Folder_Path_convert = 'D:/MASA/convert.excel'
    Folder_Path_result = 'D:/MASA/result/'
    raster_factors = open_raster(Folder_Path_factor)
    raster_main = open_raster(Folder_Path_main)
    profile = (rasterio.open('D:/MASA/main_raster/land_use.tif')).profile
    develop = input_excel(Folder_Path_develop_room)
    convert = input_excel(Folder_Path_convert)
    city_factors = [0.11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    rural_factors = [1, 0.2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    arable_factors = [1, 2, 0.3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    forest_factors = [1, 2, 3, 0.4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    grassland_factors = [1, 2, 3, 4, 0.5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    orchard_factors = [1, 2, 3, 4, 5, 0.6, 7, 8, 9, 10, 11, 12, 13, 14]
    influence_list = {
        'city': city_factors,
        'rural': rural_factors,
        'arable': arable_factors,
        'forest': forest_factors,
        'grassland': grassland_factors,
        'orchard': orchard_factors
    }
    parameter_list = {
        'optimal_len': 5,
        'number_region': 28,
        'number_year': 15,
        'develop_region': develop,
        'convert_index': convert,
        "raster_profile": profile,
        "Path_result": Folder_Path_result,
        "Path_main_data": Folder_Path_main,
        "ratio": [0.4, 0.3, 0.4]
    }
    limit_list = {
        'traffic': 18,
        'urban': 18,
        'water': 18,
        'unused': 18,
        'rural': 18,
        'indus': 18,
        'arable': 18,
        'grass': 18,
        'forest': 18,
        'orchard': 18}
    land_use_code_put = {
        'traffic': 1,
        'urban': 2,
        'water': 3,
        'unused': 4,
        'rural': 5,
        'indus': 6,
        'arable': 7,
        'grassland': 8,
        'forest': 9,
        'orchard': 10
    }
    government_agent(raster_factors, raster_main, parameter_list, limit_list, influence_list,
                     land_use_code_put)
