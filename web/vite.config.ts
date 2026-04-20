import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies /ws/* and /api/* to the FastAPI backend on :8000.
// In production we build to dist/ and FastAPI serves it, so this is dev-only.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
