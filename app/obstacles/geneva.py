"""Surveyed Geneva rooftop superstructures in LV95, with complete pagination."""
from shapely.geometry import shape

SERVICE = "https://vector.sitg.ge.ch/arcgis/rest/services/CAD_BATIMENT_HORSOL_TOIT_SP/FeatureServer/0/query"
EXTENT = (2486335, 1110440, 2512591, 1135439)


async def superstructures(client, bbox):
    if bbox[2] < EXTENT[0] or bbox[0] > EXTENT[2] or bbox[3] < EXTENT[1] or bbox[1] > EXTENT[3]:
        return []
    features = []
    for offset in range(0, 200000, 4000):
        response = await client.get(SERVICE, params={
            "f": "geojson", "where": "1=1", "geometry": ",".join(map(str, bbox)),
            "geometryType": "esriGeometryEnvelope", "inSR": 2056, "outSR": 2056,
            "outFields": "OBJECTID,EGID,ALTITUDE_MIN,ALTITUDE_MAX,DATE_LEVE",
            "orderByFields": "OBJECTID", "resultOffset": offset, "resultRecordCount": 4000})
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError("Geneva superstructure service returned an error")
        if data.get("crs", {}).get("properties", {}).get("name") != "EPSG:2056":
            raise ValueError("Geneva service returned an unexpected coordinate system")
        page = data.get("features", [])
        features.extend(page)
        if not (data.get("exceededTransferLimit") or data.get("properties", {}).get("exceededTransferLimit")):
            return features
        if not page:
            raise ValueError("Geneva pagination stopped before all labels were returned")
    raise ValueError("Geneva query exceeded safe pagination limit")


def surveyed_polygons(features, roof):
    for feature in features:
        geometry = shape(feature["geometry"]).buffer(0).intersection(roof)
        parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        for part in parts:
            if part.geom_type == "Polygon" and part.area >= .15:
                yield part, feature.get("properties", {})
