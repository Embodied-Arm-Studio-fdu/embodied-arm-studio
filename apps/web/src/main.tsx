import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "./App";
import { I18nProvider, LocalizedContent } from "./i18n";
import "./styles.css";

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: 1 } } });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <I18nProvider>
      <LocalizedContent>
        <QueryClientProvider client={queryClient}>
          <App />
        </QueryClientProvider>
      </LocalizedContent>
    </I18nProvider>
  </React.StrictMode>,
);