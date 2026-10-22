import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { Decoration, EditorView, type ViewUpdate } from "@codemirror/view";
import { useCallback, useMemo, useRef } from "react";

type Props = {
  locale?: "zh-CN" | "en";
  value: string;
  readOnly: boolean;
  original?: string;
  onChange?: (value: string) => void;
  onSelection?: (value: string) => void;
};

const extensions = [python(), EditorView.lineWrapping, EditorView.theme({
    "&": { fontSize: "13px", height: "100%" },
    ".cm-content": { fontFamily: '"IBM Plex Mono", monospace', padding: "16px 0" },
    ".cm-scroller": { overflow: "auto" },
    ".cm-gutters": { background: "#f4f7f5", borderRight: "1px solid #d8e1db" },
})];

export function WorkbenchEditor({ value, readOnly, original, onChange, onSelection, locale = "zh-CN" }: Props) {
  const view = useRef<EditorView | null>(null);
  const changeCallback = useRef(onChange);
  const selectionCallback = useRef(onSelection);
  changeCallback.current = onChange;
  selectionCallback.current = onSelection;
  // Keep the editor's listeners and extensions stable while React updates the
  // teaching cards. Reconfiguration during native input can interrupt editing.
  const handleChange = useCallback((next: string) => changeCallback.current?.(next), []);
  const handleUpdate = useCallback((update: ViewUpdate) => {
    if (update.selectionSet) {
      const range = update.state.selection.main;
      selectionCallback.current?.(update.state.sliceDoc(range.from, range.to));
    }
  }, []);
  const editorExtensions = useMemo(() => readOnly ? extensions : [...extensions, EditorView.decorations.compute(["doc"], (state) => {
    const lines = [];
    for (let number = 1; number <= state.doc.lines; number++) {
      const line = state.doc.line(number);
      if (line.text.trimStart().startsWith("return ")) lines.push(Decoration.line({ class: "workbench-editable-line" }).range(line.from));
    }
    return Decoration.set(lines);
  })], [readOnly]);
  function scrollToEdge(bottom: boolean) {
    if (view.current) view.current.dispatch({ effects: EditorView.scrollIntoView(bottom ? view.current.state.doc.length : 0, { y: bottom ? "end" : "start" }) });
  }
  if (original !== undefined) return <div className="workbench-native-editor workbench-diff" translate="no">
    <section><strong>{locale === "en" ? "BASELINE" : "基线版本"}</strong><CodeMirror value={original} readOnly editable={false} height="100%" extensions={extensions} /></section>
    <section><strong>{locale === "en" ? "CURRENT" : "当前版本"}</strong><CodeMirror value={value} readOnly editable={false} height="100%" extensions={extensions} /></section>
  </div>;
  return <div className="workbench-native-editor workbench-scrollable-editor" translate="no">
    <div className="workbench-scroll-tools"><span>{locale === "en" ? `${value.split("\n").length} lines · Scroll within the editor` : `${value.split("\n").length} 行 · 在代码区上下滚动`}</span><button type="button" onClick={() => scrollToEdge(false)}>{locale === "en" ? "Top" : "顶部"}</button><button type="button" onClick={() => scrollToEdge(true)}>{locale === "en" ? "End of file" : "文件末尾"}</button></div>
    <CodeMirror value={value} autoFocus={!readOnly} height="100%" style={{ flex: 1, minHeight: 0, overflow: "hidden" }} onCreateEditor={(editor) => { view.current = editor; }} readOnly={readOnly} editable={!readOnly} extensions={editorExtensions} onChange={onChange ? handleChange : undefined} onUpdate={handleUpdate} /></div>;
}
