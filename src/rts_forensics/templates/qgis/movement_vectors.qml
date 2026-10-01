<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology">
  <!-- Arrow geometry-generator style for the movement_vectors layer.
       The displayed length is scaled by the project variable @hlo_vec_scale.
       Documented default from config.gis.vector_scale: 2500. -->
  <renderer-v2 type="singleSymbol" forceraster="0" symbollevels="0"
               enableorderby="0" referencescale="-1">
    <symbols>
      <symbol type="line" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="GeometryGenerator" enabled="1" pass="0" locked="0" id="arrow">
          <prop k="SymbolType" v="Line"/>
          <prop k="geometryModifier" v="make_line(start_point($geometry), start_point($geometry) + coalesce(@hlo_vec_scale, 2500) * (end_point($geometry) - start_point($geometry)))"/>
          <Option type="Map">
            <Option name="SymbolType" type="QString" value="Line"/>
            <Option name="geometryModifier" type="QString" value="make_line(start_point($geometry), start_point($geometry) + coalesce(@hlo_vec_scale, 2500) * (end_point($geometry) - start_point($geometry)))"/>
          </Option>
          <symbol type="line" name="arrow_line" alpha="1" clip_to_extent="1" force_rhr="0">
            <layer class="SimpleLine" enabled="1" pass="0" locked="0" id="shaft">
              <prop k="line_color" v="35,35,35,255"/>
              <prop k="line_width" v="0.4"/>
            </layer>
            <layer class="MarkerLine" enabled="1" pass="0" locked="0" id="head">
              <prop k="placement" v="lastvertex"/>
              <prop k="symbol" v="arrow_head"/>
            </layer>
          </symbol>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerGeometryType>1</layerGeometryType>
</qgis>
