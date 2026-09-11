import { sites } from '@openai/sites-vite-plugin';
import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig } from 'vite';
// scripts/serve.py picks free ports and passes them in; defaults match a manual run.
const webPort = Number(process.env.PV_WEB_PORT ?? 5173);
const apiPort = Number(process.env.PV_API_PORT ?? 8000);
export default defineConfig({
  css: { postcss: { plugins: [tailwindcss()] } },
  server: { host: '127.0.0.1', port: webPort, strictPort: true,
    allowedHosts: ['renewably-lily-educator.ngrok-free.dev'],
    proxy: { '/api': { target: `http://127.0.0.1:${apiPort}`, timeout: 180000 } } },
  plugins: [vinext(), sites()],
});
