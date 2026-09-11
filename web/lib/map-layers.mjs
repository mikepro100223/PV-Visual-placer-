export const CITY_LABEL_MAX_ZOOM = 14;

export const SWISS_AERIAL_URL =
  'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg';

export const SWISS_CITY_CONTEXT_URL =
  'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg';

/** @param {number} zoom */
export function cityLabelsAreVisible(zoom) {
  return zoom <= CITY_LABEL_MAX_ZOOM;
}
