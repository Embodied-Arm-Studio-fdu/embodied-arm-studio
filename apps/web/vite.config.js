var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(function (_a) {
    var _b, _c;
    var mode = _a.mode;
    var environment = __assign(__assign({}, loadEnv(mode, ".", "")), process.env);
    var proxyTarget = (_b = environment.VITE_API_PROXY_TARGET) !== null && _b !== void 0 ? _b : "http://127.0.0.1:8000";
    console.info("[vite] API proxy target: ".concat(proxyTarget));
    return {
        plugins: [react()],
        server: {
            port: 5173,
            proxy: {
                "/media-whep": {
                    target: (_c = environment.VITE_WHEP_PROXY_TARGET) !== null && _c !== void 0 ? _c : "http://127.0.0.1:18889",
                    changeOrigin: true,
                    rewrite: function (path) { return path.replace(/^\/media-whep/, ""); },
                    configure: function (proxy) {
                        proxy.on("proxyReq", function (request) { return request.removeHeader("origin"); });
                        proxy.on("proxyRes", function (response) {
                            var location = response.headers.location;
                            if (location) {
                                var target = new URL(location, "http://127.0.0.1:18889");
                                response.headers.location = "/media-whep".concat(target.pathname).concat(target.search);
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
