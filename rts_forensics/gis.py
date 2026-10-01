"""Ten-layer GeoPackage with default QGIS styles, including browser support."""

import html
import math

import numpy as np
import pandas as pd

from prismacore.gpkg_lite import GpkgWriter


def transform_xy_vectors(x, y, ve, vn, transform=None):
    if not transform:
        return x, y, ve, vn
    theta = np.radians(transform.get("rotation_deg", 0))
    c, s = np.cos(theta), np.sin(theta)
    return (
        c * x - s * y + transform.get("offset_e", 0),
        s * x + c * y + transform.get("offset_n", 0),
        c * ve - s * vn,
        s * ve + c * vn,
    )


def _clean(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def qml(name, geometry):
    marker = "line" if geometry == "LINESTRING" else "fill" if geometry == "POLYGON" else "marker"
    symbol = "SimpleLine" if marker == "line" else "SimpleFill" if marker == "fill" else "SimpleMarker"
    temporal = (
        '<temporal enabled="1" mode="0" startField="block_start" endField="block_end"/>'
        if name == "timeseries_24h"
        else ""
    )
    generator = ""
    if name == "prism_summary":
        expression = 'make_line($geometry, translate($geometry, "vector_e_mm" / 1000 * coalesce(@hlo_vec_scale,1000), "vector_n_mm" / 1000 * coalesce(@hlo_vec_scale,1000)))'
        generator = f'<layer class="GeometryGenerator" enabled="1"><Option type="Map"><Option name="geometryModifier" value="{html.escape(expression, quote=True)}" type="QString"/><Option name="SymbolType" value="Line" type="QString"/></Option><symbol type="line" name="@0@1"><layer class="SimpleLine"><Option type="Map"><Option name="line_color" value="255,140,0,255" type="QString"/></Option></layer></symbol></layer>'
    return f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
<renderer-v2 type="singleSymbol"><symbols><symbol type="{marker}" name="0">
<layer class="{symbol}" enabled="1"><Option type="Map"><Option name="color" value="40,120,180,255" type="QString"/><Option name="line_color" value="40,120,180,255" type="QString"/></Option></layer>{generator}
</symbol></symbols></renderer-v2>{temporal}
<fieldConfiguration><field name="status"><editWidget type="ValueMap"><config><Option type="Map"><Option name="map" type="List"><Option type="Map"><Option name="Open" value="open" type="QString"/><Option name="Completed" value="completed" type="QString"/></Option></Option></config></editWidget></field><field name="photo"><editWidget type="ExternalResource"><config/></editWidget></field></fieldConfiguration>
<editform tolerant="1"/><editorlayout>generatedlayout</editorlayout>
<customproperties><Option type="Map"><Option name="forensics/notice" value="Exploratory; raw and corrected side by side; uncertainty is frame fit only" type="QString"/></Option></customproperties>
</qgis>'''


class StyledWriter(GpkgWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.styles = []

    def add_styled_layer(self, name, geometry, rows):
        super().add_layer(name, geometry, rows, description="Exploratory RTS network forensics")
        if name not in self._nama_terpakai:
            self._layer.append(
                {
                    "nama": name,
                    "geom_key": "geom",
                    "geom_type": geometry,
                    "kolom": ["status"],
                    "tipe": ["TEXT"],
                    "baris": [],
                    "deskripsi": "No supported features in this run",
                    "env": (None, None, None, None),
                }
            )
            self._nama_terpakai.add(name)
        self.styles.append(
            (
                None,
                "",
                "",
                name,
                "geom",
                name,
                qml(name, geometry),
                "",
                1,
                "Default exploratory style",
                "rts-forensics",
            )
        )

    def _skema(self):
        schema = super()._skema()
        cols = [
            "id",
            "f_table_catalog",
            "f_table_schema",
            "f_table_name",
            "f_geometry_column",
            "styleName",
            "styleQML",
            "styleSLD",
            "useAsDefault",
            "description",
            "owner",
        ]
        types = ["INTEGER"] + ["TEXT"] * 7 + ["INTEGER", "TEXT", "TEXT"]
        ddl = (
            "CREATE TABLE layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            + ", ".join(f'"{c}" {t}' for c, t in zip(cols[1:], types[1:]))
            + ")"
        )
        schema.append(
            {
                "nama": "layer_styles",
                "ddl": ddl,
                "kolom": cols,
                "tipe": types,
                "rowid_kolom": 0,
                "autoincrement": True,
                "indeks": [],
                "baris": self.styles,
            }
        )
        return schema


def write_gpkg(result, path, backend="auto"):
    transform = result["config"]["site_transform"]
    srs = transform.get("srs_id", -1) if transform else -1
    wkt = transform.get("srs_wkt") if transform else None
    if srs not in [-1, 0] and not wkt:
        raise ValueError("A transformed CRS needs an explicit srs_wkt; no CRS is inferred")
    points, stations, vectors, artefacts, sightlines = [], [], [], [], []
    summary = result["prism_summary"]
    coordinates = {}
    station_coordinates = {}
    for r in result["cycle_table"].groupby("station").first().reset_index().itertuples():
        x, y, _, _ = transform_xy_vectors(r.station_e, r.station_n, 0, 0, transform)
        station_coordinates[r.station] = (x, y)
        stations.append(
            {
                "station": r.station,
                "geom": (x, y),
                "input_grid": "configured" if transform else "undefined Cartesian",
            }
        )
    for _, r in summary.iterrows():
        x, y, ve, vn = transform_xy_vectors(r.e, r.n, r.vector_e_mm, r.vector_n_mm, transform)
        row = {k: _clean(v) for k, v in r.to_dict().items() if k != "source_refs"}
        row.update(vector_e_mm=_clean(ve), vector_n_mm=_clean(vn), geom=(x, y))
        points.append(row)
        coordinates[(r.station, r.pid)] = (x, y)
        if np.isfinite(ve) and np.isfinite(vn):
            vectors.append(
                {
                    "pid": r.pid,
                    "station": r.station,
                    "vector_e_mm": ve,
                    "vector_n_mm": vn,
                    "experimental": True,
                    "los_fit_se_mm": _clean(r.los_fit_se_mm),
                    "geom": [(x, y), (x + ve / 1000, y + vn / 1000)],
                }
            )
        g = result["series"][(result["series"].pid == r.pid) & (result["series"].station == r.station)]
        angle = np.radians(r.az_deg)
        effect = g.tan_raw.tail(1).iloc[0] - g.tan_fc.tail(1).iloc[0]
        _, _, ae, an = transform_xy_vectors(0, 0, effect * np.cos(angle), -effect * np.sin(angle), transform)
        if np.isfinite(ae) and np.isfinite(an):
            artefacts.append(
                {"pid": r.pid, "experimental": True, "geom": [(x, y), (x + ae / 1000, y + an / 1000)]}
            )
        sightlines.append(
            {
                "pid": r.pid,
                "station": r.station,
                "lost_final_48h": bool(r.lost_final_48h),
                "geom": [station_coordinates[r.station], (x, y)],
            }
        )
    temporal = []
    for r in result["timeseries_24h"].to_dict("records"):
        temporal.append(
            {
                **{k: _clean(v) for k, v in r.items() if k != "source_refs"},
                "geom": coordinates[(r["station"], r["pid"])],
            }
        )
    # Global checks are anchored at an instrument location, with explicit scope.
    default_point = next(iter(station_coordinates.values()))

    def located(table):
        records = []
        for r in result[table].to_dict("records"):
            key = (r.get("station"), r.get("pid", r.get("prisms")))
            geom = coordinates.get(key, station_coordinates.get(r.get("station"), default_point))
            records.append(
                {
                    **{k: _clean(v) for k, v in r.items() if k != "source_refs"},
                    "geometry_scope": "target" if key in coordinates else "network/station",
                    "geom": geom,
                }
            )
        return records

    # No guessed zone extent: configured cluster radius defines candidate zones.
    zones = []
    radius = result["config"]["detection"]["cluster_radius_m"]
    if radius:
        for r in points:
            if r["movement_concern"].startswith("detected"):
                x, y = r["geom"]
                ring = [
                    (x + radius * np.cos(a), y + radius * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 33)
                ]
                zones.append({"pid": r["pid"], "radius_m": radius, "exploratory": True, "geom": ring})
    with StyledWriter(path, srs_id=srs, srs_wkt=wkt, backend=backend) as writer:
        for name, geometry, rows in [
            ("prism_summary", "POINT", points),
            ("stations", "POINT", stations),
            ("movement_vectors", "LINESTRING", vectors),
            ("artefact_vectors", "LINESTRING", artefacts),
            ("sight_lines", "LINESTRING", sightlines),
            ("candidate_zones", "POLYGON", zones),
            ("timeseries_24h", "POINT", temporal),
            ("events", "POINT", located("event_register")),
            ("investigations", "POINT", located("investigation_register")),
            ("field_checks", "POINT", located("field_checks")),
        ]:
            writer.add_styled_layer(name, geometry, rows)
    return path
