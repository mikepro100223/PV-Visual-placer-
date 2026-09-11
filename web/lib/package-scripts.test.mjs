import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const packageJson = JSON.parse(
  await readFile(new URL('../package.json', import.meta.url), 'utf8'),
);

void test('the production start command uses the Vinext build output', () => {
  assert.equal(packageJson.scripts.start, 'vinext start');
});
