import os.path

import geopandas as gpd
import osmnx as ox
from pyproj import CRS
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt  # 确保导入了这个模块
import networkx as nx
import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

# 定义城市列表
cities = [
    "Beijing, China", "Tokyo, Japan", "New York, USA", "London, UK", "Paris, France",
    "Moscow, Russia", "Sydney, Australia", "Berlin, Germany", "Mumbai, India", "Sao Paulo, Brazil",
    "Mexico City, Mexico", "Cairo, Egypt", "Bangkok, Thailand", "Buenos Aires, Argentina", "Tehran, Iran",
    "Istanbul, Turkey", "Jakarta, Indonesia", "Seoul, South Korea", "Kinshasa, DR Congo", "Lagos, Nigeria",
    "Shanghai, China", "Los Angeles, USA", "Kolkata, India", "Manila, Philippines", "Toronto, Canada",
    "Rio de Janeiro, Brazil", "Guangzhou, China", "Lima, Peru", "Cape Town, South Africa", "Madrid, Spain",
    "Melbourne, Australia", "Nairobi, Kenya", "Singapore, Singapore", "Chennai, India", "Houston, USA",
    "Saint Petersburg, Russia", "Lahore, Pakistan", "Baghdad, Iraq", "Santiago, Chile", "Bangalore, India",
    "Hyderabad, India", "Quito, Ecuador", "Montreal, Canada", "Caracas, Venezuela", "Hanoi, Vietnam",
    "Hong Kong, China", "Bogota, Colombia", "Dhaka, Bangladesh", "Riyadh, Saudi Arabia", "Barcelona, Spain"
]


def draw_plath(place_name, save_path):
    save_path = os.path.join(save_path, place_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gdf = ox.geocode_to_gdf(place_name)
    chaoyang_polygon = gdf.loc[0, 'geometry']  # 获取多边形

    # 将多边形转换到适合其位置的 UTM 坐标系
    utm_crs = CRS(ox.projection.project_geometry(chaoyang_polygon)[1])
    gdf_utm = gdf.to_crs(utm_crs)
    chaoyang_polygon_utm = gdf_utm.loc[0, 'geometry']
    # utm 坐标系

    # 定义网格大小（米）
    grid_size_m = 5000

    # 计算 UTM 投影后的多边形边界
    minx, miny, maxx, maxy = chaoyang_polygon_utm.bounds

    # 创建正方形网格覆盖多边形
    x_coords = np.arange(minx, maxx, grid_size_m)
    y_coords = np.arange(miny, maxy, grid_size_m)
    grid_polygons = []
    for x in x_coords:
        for y in y_coords:
            # 每个正方形的边界
            square = box(x, y, x + grid_size_m, y + grid_size_m)
            # 如果正方形和多边形相交，则加入到列表中
            if square.intersects(chaoyang_polygon_utm):
                grid_polygons.append(square.intersection(chaoyang_polygon_utm))

    # 创建GeoDataFrame以便于可视化和进一步处理
    grid_gdf_utm = gpd.GeoDataFrame(grid_polygons, columns=['geometry'], crs=utm_crs)

    # 如需可视化，需要将其转换回原始 CRS
    grid_gdf_wgs = grid_gdf_utm.to_crs(gdf.crs)

    # 可视化
    # ax = gdf.plot(edgecolor='blue', facecolor='none')
    # grid_gdf_wgs.plot(ax=ax, edgecolor='red', facecolor='none')
    # plt.show()  # 显示图像
    area_threshold = 1000
    index = 0
    for polygon in grid_gdf_wgs['geometry']:
        try:
            index = index + 1
            # 使用custom_filter参数过滤主干道，比如 '["highway"~"motorway|trunk"]' 只获取高速公路和干线
            main_road = ox.graph_from_polygon(polygon, network_type='all',
                                              custom_filter='["highway"~"motorway|trunk|primary"]')
            fig, ax = ox.plot_graph(main_road, node_size=0, save=True, show=False,
                                    filepath=os.path.join(save_path, f"main_{index}.png"), close=True)

            ###################################################################################################

            graph = ox.graph_from_polygon(polygon, network_type='all')
            # 绘制路网图，设置图形的边界为多边形的边界
            fig, ax = ox.plot_graph(graph, node_size=0, save=True, show=False,
                                    filepath=os.path.join(save_path, f"all_{index}.png"))

        except ValueError:
            # print("ValueError")
            continue
        except nx.NetworkXPointlessConcept:
            # 如果图是空的，跳过这个多边形
            continue


if __name__ == "__main__":
    save_path = "res"
    with ProcessPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(draw_plath, city, save_path) for city in cities]

    # 等待所有任务完成
    for future in futures:
        future.result()

    # draw_plath(place_name, save_path)
