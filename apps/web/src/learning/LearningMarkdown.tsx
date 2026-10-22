import { useI18n } from "../i18n";
import { Fragment, useState, type ReactNode } from "react";
import { Check, Copy } from "lucide-react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { LearningEmbeddedVisual } from "./RoboticsDiagrams";

function mathMarkup(source: string, displayMode: boolean): string {
  try {
    return katex.renderToString(source, { displayMode, throwOnError: false, strict: "warn" });
  } catch {
    return source;
  }
}

function inline(text: string): ReactNode[] {
  const pattern = /(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|\*[^*]+\*|\\\([\s\S]*?\\\))/g;
  return text.split(pattern).filter(Boolean).map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (link) return <a key={index} href={link[2]} target="_blank" rel="noreferrer">{link[1]}</a>;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*")) return <em key={index}>{part.slice(1, -1)}</em>;
    if (part.startsWith("\\(") && part.endsWith("\\)")) return <span className="learning-math-inline" key={index} dangerouslySetInnerHTML={{ __html: mathMarkup(part.slice(2, -2), false) }}/>;
    return <Fragment key={index}>{part}</Fragment>;
  });
}

function isBoundary(line: string): boolean {
  return /^(### |```|\\\[|>|[-*] |\d+\. |\||<details>|<\/details>|<summary>|:::learning-visual)/.test(line);
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(code);
    } else if (typeof document !== "undefined") {
      const input = document.createElement("textarea");
      input.value = code;
      input.style.position = "fixed";
      input.style.opacity = "0";
      document.body.appendChild(input);
      input.select();
      document.execCommand("copy");
      input.remove();
    }
    setCopied(true);
    if (typeof window !== "undefined") window.setTimeout(() => setCopied(false), 1600);
  };
  return <figure className="learning-code"><figcaption><span>{language} · {language === "python" ? t("只读教学示例（不在浏览器执行）", "Read-only teaching example (not executed in the browser)") : t("代码示例", "Code example")}</span><button type="button" onClick={() => void copy()} aria-label={t("复制代码", "Copy code")}>{copied ? <Check size={13}/> : <Copy size={13}/>} {copied ? t("已复制", "Copied") : t("复制", "Copy")}</button></figcaption><pre><code>{code}</code></pre></figure>;
}

function detailsClass(summary: string): string {
  if (/平台真实实现|Platform implementation/i.test(summary)) return "learning-details implementation";
  if (/进阶|Advanced/i.test(summary)) return "learning-details advanced";
  return "learning-details";
}

export function LearningMarkdown({ source }: { source: string }) {
  const { t } = useI18n();
  const lines = source.split("\n");
  const output: ReactNode[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trimEnd();
    if (!line.trim()) { index += 1; continue; }
    if (line.startsWith("```")) {
      const language = line.slice(3).trim() || "text";
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) code.push(lines[index++]);
      index += 1;
      output.push(<CodeBlock language={language} code={code.join("\n")} key={`code-${index}`}/>);
      continue;
    }
    const visual = line.trim().match(/^:::learning-visual\s+([a-z-]+)$/);
    if (visual) {
      output.push(<LearningEmbeddedVisual kind={visual[1]} key={`visual-${index}`}/>);
      index += 1;
      continue;
    }
    if (line.trim() === "<details>") {
      const detailLines: string[] = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== "</details>") detailLines.push(lines[index++]);
      index += 1;
      const summaryIndex = detailLines.findIndex((item) => item.trim().startsWith("<summary>"));
      const summary = summaryIndex >= 0 ? detailLines[summaryIndex].trim().replace(/^<summary>/, "").replace(/<\/summary>$/, "") : t("展开阅读", "Read more");
      const body = detailLines.filter((_, lineIndex) => lineIndex !== summaryIndex).join("\n").trim();
      output.push(<details className={detailsClass(summary)} key={`details-${index}`}><summary>{summary}</summary><div><LearningMarkdown source={body}/></div></details>);
      continue;
    }
    if (line === "\\[") {
      const formula: string[] = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== "\\]") formula.push(lines[index++].trim());
      index += 1;
      output.push(<div className="learning-equation" role="math" key={`math-${index}`} dangerouslySetInnerHTML={{ __html: mathMarkup(formula.join(" "), true) }}/>);
      continue;
    }
    if (line.startsWith("|")) {
      const rows: string[][] = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) rows.push(lines[index++].trim().slice(1, -1).split("|").map((cell) => cell.trim()));
      const [header, divider, ...body] = rows;
      if (divider?.every((cell) => /^:?-+:?$/.test(cell))) output.push(<div className="learning-table-wrap" key={`table-${index}`}><table><thead><tr>{header.map((cell, cellIndex) => <th key={cellIndex}>{inline(cell)}</th>)}</tr></thead><tbody>{body.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{inline(cell)}</td>)}</tr>)}</tbody></table></div>);
      continue;
    }
    if (line.startsWith(">")) {
      const quote: string[] = [];
      while (index < lines.length && lines[index].trim().startsWith(">")) quote.push(lines[index++].trim().replace(/^>\s?/, ""));
      output.push(<aside className="learning-callout" key={`quote-${index}`}>{quote.map((item, quoteIndex) => item ? <p key={quoteIndex}>{inline(item.replace(/^[-*]\s+/, ""))}</p> : null)}</aside>);
      continue;
    }
    if (/^[-*] /.test(line) || /^\d+\. /.test(line)) {
      const ordered = /^\d+\. /.test(line);
      const items: string[] = [];
      const matcher = ordered ? /^\d+\.\s+/ : /^[-*]\s+/;
      while (index < lines.length && matcher.test(lines[index].trim())) items.push(lines[index++].trim().replace(matcher, ""));
      const List = ordered ? "ol" : "ul";
      output.push(<List key={`list-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{inline(item)}</li>)}</List>);
      continue;
    }
    if (line.startsWith("### ")) {
      output.push(<h3 key={`heading-${index}`}>{line.slice(4)}</h3>);
      index += 1;
      continue;
    }
    if (/^(参考解释|参考答案|参考实现|答案是|Reference (?:answer|explanation|approach|implementation):|The answer is)/i.test(line.trim())) {
      const answer = [line.trim()];
      index += 1;
      while (index < lines.length && lines[index].trim() && !isBoundary(lines[index].trim())) answer.push(lines[index++].trim());
      output.push(<details className="learning-answer" key={`answer-${index}`}><summary>{t("查看参考答案", "Show reference answer")}</summary><p>{inline(answer.join(" "))}</p></details>);
      continue;
    }
    const paragraph = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isBoundary(lines[index].trim())) paragraph.push(lines[index++].trim());
    const paragraphText = paragraph.join(" ");
    const inlineAnswer = paragraphText.match(/^(\*\*(?:预测|参数|数值|修改)练习。?\*\*.*?)(参考解释：|参考答案：)(.+)$/);
    if (inlineAnswer) {
      output.push(<p className="learning-body-paragraph" key={`paragraph-${index}`}>{inline(inlineAnswer[1].trim())}</p>);
      output.push(<details className="learning-answer" key={`inline-answer-${index}`}><summary>{t("查看参考答案", "Show reference answer")}</summary><p>{inline(`${inlineAnswer[2]}${inlineAnswer[3]}`)}</p></details>);
    } else {
      output.push(<p className="learning-body-paragraph" key={`paragraph-${index}`}>{inline(paragraphText)}</p>);
    }
  }
  return <>{output}</>;
}

