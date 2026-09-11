import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CITY_LABEL_MAX_ZOOM,
  SWISS_AERIAL_URL,
  SWISS_CITY_CONTEXT_URL,
  cityLabelsAreVisible,
} from './map-layers.mjs';

void test('city labels are limited to the large zoomed-out map view', () => {
  assert.equal(cityLabelsAreVisible(CITY_LABEL_MAX_ZOOM - 1), true);
  assert.equal(cityLabelsAreVisible(CITY_LABEL_MAX_ZOOM), true);
  assert.equal(cityLabelsAreVisible(CITY_LABEL_MAX_ZOOM + 1), false);
});

void test('both zoom modes use official swisstopo tiles', () => {
  assert.match(SWISS_AERIAL_URL, /ch\.swisstopo\.swissimage/);
  assert.match(SWISS_CITY_CONTEXT_URL, /ch\.swisstopo\.pixelkarte-farbe/);
});
