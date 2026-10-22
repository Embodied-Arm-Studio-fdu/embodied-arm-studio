import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CopyableIdentifier } from "./App";
import { I18nProvider } from "./i18n";

describe("CopyableIdentifier", () => {
  it("keeps the complete identifier in selectable DOM text", () => {
    const value = "3f01b838-1428-4f1d-997a-d8ea72338a9a";
    const markup = renderToStaticMarkup(
      <I18nProvider>
        <CopyableIdentifier value={value} label="Run ID" />
      </I18nProvider>,
    );

    expect(markup).toContain(`>${value}</span>`);
    expect(markup).not.toContain("3f01b838...338a9a");
  });
});
