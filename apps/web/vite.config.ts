import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

declare const process: {
  env: Record<string, string | undefined>;
};

export default defineConfig(({ mode }) => {
  const environment = { ...loadEnv(mode, ".", ""), ...process.env };
  const proxyTarget =
    environment.VITE_API_PROXY_TARGET ??
    "http://127.0.0.1:8000";

  console.info(`[vite] API proxy target: ${proxyTarget}`);

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/media-whep": {
          target: environment.VITE_WHEP_PROXY_TARGET ?? "http://127.0.0.1:18889",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/media-whep/, ""),
          configure: (proxy) => {
            proxy.on("proxyReq", (request) => request.removeHeader("origin"));
            proxy.on("proxyRes", (response) => {
              const location = response.headers.location;
              if (location) {
                const target = new URL(location, "http://127.0.0.1:18889");
                response.headers.location = `/media-whep${target.pathname}${target.search}`;
              }
            });
          },
        },
        "/api": { target: proxyTarget, ws: true },
        "/ws": { target: proxyTarget.replace(/^http:/, "ws:"), ws: true },
      },
    },
  };
});