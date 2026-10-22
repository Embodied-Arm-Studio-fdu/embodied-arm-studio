import { describe, expect, it } from "vitest";
import factory from "./vite.config";

describe("same-origin WHEP development proxy", () => {
  it("rewrites session Location with query and leaves the browser port intact", () => {
    const config = (factory as Function)({ mode: "test", command: "serve" });
    const proxy = config.server.proxy["/media-whep"];
    const handlers: Record<string, Function> = {};
    proxy.configure({ on: (event: string, handler: Function) => { handlers[event] = handler; } });
    for (const location of ["http://media:18889/stream/session?token=test", "/stream/session?token=test"]) {
      const response = { headers: { location } };
      handlers.proxyRes(response);
      expect(response.headers.location).toBe("/media-whep/stream/session?token=test");
      expect(new URL(response.headers.location, "http://localhost:18083").port).toBe("18083");
    }
    expect(proxy.rewrite("/media-whep/stream/whep")).toBe("/stream/whep");
    expect(config.server.proxy["/media-hls"]).toBeUndefined();
  });
});
