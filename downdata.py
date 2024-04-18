import osmnx as ox
import geopandas as gpd
import os
import networkx as nx

data_path = 'oridata'


def download_and_save_grid_road_data(city_name, grid_size, filename_base):
    """
    Download and save detailed road data for each grid section of the city.

    :param city_name: Name of the city to download the data for.
    :param grid_size: The number of grid divisions along one axis.
    :param filename_base: Base name for saving each grid image.
    """
    # 获取城市边界
    city = ox.geocode_to_gdf(city_name)
    city_boundary = city.unary_union
    minx, miny, maxx, maxy = city_boundary.bounds

    # 计算网格的宽度和高度
    dx = (maxx - minx) / grid_size
    dy = (maxy - miny) / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            # 计算当前网格的边界
            north = maxy - j * dy
            south = maxy - (j + 1) * dy
            west = minx + i * dx
            east = minx + (i + 1) * dx

            # 使用bbox参数下载并保存网格内的路网数据
            bbox = (north, south, east, west)
            G = ox.graph_from_bbox(bbox=bbox, network_type='all', simplify=False, retain_all=True)
            nx.write_graphml(G, os.path.join(data_path, f"{filename_base}_grid_{i}_{j}.graphml"))

            # nodes, edges = ox.graph_to_gdfs(G)
            #
            # nodes.to_file(os.path.join(data_path, f"{filename_base}_grid_{i}_{j}_nodes.gpkg"), driver='GPKG')
            # edges.to_file(os.path.join(data_path, f"{filename_base}_grid_{i}_{j}_edges.gpkg"), driver='GPKG')

            print(f"Data for grid {i},{j} saved.")


# 下载并保存城市网格化路网数据
download_and_save_grid_road_data('Beijing, China', grid_size=10, filename_base='beijing_road_data')
