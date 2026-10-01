# QGIS styles

`movement_vectors.qml` is the default arrow style for the `movement_vectors`
GeoPackage layer. It uses a geometry generator that draws an arrow from the
start point to the vector tip, scaled by the project variable
`@hlo_vec_scale` (documented default `2500`, matching
`config.gis.vector_scale`). Override the project variable to change the arrow
length without editing the layer style.

`rts_forensics.gis.write_qml` writes the same style with the configured
`config.gis.vector_scale` substituted as the default. Candidate zones follow
an explicit configured spatial rule; without one the layer is exported empty
with a `not_configured` reason. Imported geometry stays in the input grid with
`srs_id = -1` unless a transform is explicitly configured.
