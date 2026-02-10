import geopandas as gpd
import movingpandas as mpd
from holoviews import Overlay
import colorcet as cc
from tqdm import tqdm

Category20_20 = (
	'#1f77b4',
	'#aec7e8',
	'#ff7f0e',
	'#ffbb78',
	'#2ca02c',
	'#98df8a',
	'#d62728',
	'#ff9896',
	'#9467bd',
	'#c5b0d5',
	'#8c564b',
	'#c49c94',
	'#e377c2',
	'#f7b6d2',
	'#7f7f7f',
	'#c7c7c7',
	'#bcbd22',
	'#dbdb8d',
	'#17becf',
	'#9edae5',
)
MPD_PALETTE = list(Category20_20) + cc.palette['glasbey']
default_width = 800
default_height = 200


def plot_trajectory_from_points(
	gdf: gpd.GeoDataFrame, hover_cols=None, legend=False, *args, **kwargs
):
	"""
	Visualise la trajectoire d'un navire en utilisant la bibliothèque moving pandas et holoviews.

	Args:
		gdf: GeoDataFrame contenant les messages AIS
		hover_cols: Liste de colonnes à afficher sur le passage de la souris
		legend: Si Vrai, affiche l'index des épisodes en tant que légende (Déconseillé pour un grand
			dataset)
		*args: arguments supplémentaires à rajouter à la fonction hvplot(), sans clé
		**kwargs: arguments supplémentaires à rajouter à la fonction hvplot(), avec clé

	Returns:
		L'image crée par la bibliothèque Holoview

	"""
	return mpd.TrajectoryCollection(
		gdf, x='lon', y='lat', t='timestamp',traj_id_col="mmsi"
	).hvplot(
		tiles='CartoLight',
		hover_cols=hover_cols,
		legend=legend,
		width=default_width,
		height=default_height,
		*args,
		**kwargs,
	)


def plot_trajectory_from_line(
	gdf: gpd.GeoDataFrame, hover_cols=None, legend=False, linewidth=2, *args, **kwargs
):
	"""
	Affiche la trajectoire des navires sous forme de ligne en utilisant la bibliothèque hvplot.
	Args:
		gdf: GeoDataFrame contenant les messages AIS
		hover_cols: Liste de colonnes à afficher sur le passage de la souris
		legend: Si Vrai, affiche l'index des épisodes en tant que légende (Déconseillé pour un grand
			dataset)
		linewidth: Taille des lignes
		*args: arguments supplémentaires à rajouter à la fonction hvplot(), sans clé
		**kwargs: arguments supplémentaires à rajouter à la fonction hvplot(), avec clé
	Returns:
		L'image crée par la bibliothèque Holoview

	"""
	# Cycle.default_cycles["default_colors"] = MPD_PALETTE
	# colormap = dict(zip(gdf["episode"], MPD_PALETTE[: len(gdf)])) #MPD_PALETTE#[:len(gdf)]
	if hover_cols is None:
		hover_cols = ['mmsi', 'episode']

	# return gdf.hvplot(
	# tiles = 'CartoLight',
	# geo = True,
	# hover_cols = hover_cols,
	# width = default_width,
	# by = 'episode',
	# height = default_height,
	# line_width = linewidth,
	# *args,
	# **kwargs,
	# )
	plots = []
	for i in tqdm(range(len(gdf))):
		tmp = gdf.iloc[i : i + 1, :]
		tmp = tmp.hvplot(
			line_width=linewidth,
			geo=True,
			color=MPD_PALETTE[i % len(MPD_PALETTE)],
			colorbar=True,
			hover_cols=hover_cols,
			legend=legend,
			*args,
			**kwargs,
		)
		plots.append(tmp)
	return Overlay(plots)


def plot_points(gdf: gpd.GeoDataFrame, hover_cols=None, legend=False, *args, **kwargs):
	"""
	Affiche la trajectoire des navires sous forme de points en utilisant la bibliothèque hvplot.
	Args:
		gdf: GeoDataFrame contenant les messages AIS
		hover_cols: Liste de colonnes à afficher sur le passage de la souris
		legend: Si vrai, affiche la légende (déconseillé pour les gros datasets)
		*args: arguments supplémentaires à rajouter à la fonction hvplot(), sans clé
		**kwargs: arguments supplémentaires à rajouter à la fonction hvplot(), avec clé
	Returns:
		L'image générée par hvplot
	"""
	if hover_cols is None:
		hover_cols = ['episode']
	return gdf.hvplot(
		tiles='CartoLight',
		geo=True,
		hover_cols=hover_cols,
		width=default_width,
		legend=legend,
		height=default_height,
		size=7,
		*args,
		**kwargs,
	)