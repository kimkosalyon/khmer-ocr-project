import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 3457,
    proxy: {
      '/render': 'http://localhost:3456',
      '/fonts': 'http://localhost:3456',
    },
  },
})
