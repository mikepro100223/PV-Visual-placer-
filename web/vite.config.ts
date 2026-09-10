import { sites } from '@openai/sites-vite-plugin';
import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig } from 'vite';
export default defineConfig({
  css: { postcss: { plugins: [tailwindcss()] } },
  server: { host: '127.0.0.1', port: 5173, strictPort: true,
    allowedHosts: ['renewably-lily-educator.ngrok-free.dev'],
    proxy: { '/api': { target: 'http://127.0.0.1:8000', timeout: 180000 } } },
  plugins: [vinext(), sites()],
});
