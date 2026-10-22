import englishManuscript from "../../../docs/introduction_to_embodied.en.md?raw";
import type { Locale } from "../../i18n";
import manuscript from "../../../docs/introduction_to_embodied.md?raw";

export interface LearningSection {
  id: string;
  title: string;
  markdown: string;
}

export interface LearningChapter {
  number: number;
  slug: string;
  title: string;
  sections: LearningSection[];
}

function sectionId(chapter: number, title: string, index: number): string {
  const numbered = title.match(/^(\d+\.\d+)/)?.[1];
  return numbered ? numbered.replace(".", "-") : `chapter-${chapter}-section-${index + 1}`;
}

function parseChapter(number: number, source: string, english = false): LearningChapter {
  const heading = source.match(new RegExp(english ? `^# Chapter ${number}\\s+(.+)$` : `^# 第${number}章\\s+(.+)$`, "m"));
  if (!heading || heading.index === undefined) throw new Error(`Learning manuscript is missing chapter ${number}`);
  const start = heading.index + heading[0].length;
  const next = source.slice(start).search(english ? /^# Chapter \d+\s+/m : /^# 第\d+章\s+/m);
  const body = source.slice(start, next < 0 ? undefined : start + next).replace(/<!--[\s\S]*?-->/g, "").trim();
  const matches = [...body.matchAll(/^##\s+(.+)$/gm)];
  const sections = matches.map((match, index) => ({
    id: sectionId(number, match[1].trim(), index),
    title: match[1].trim(),
    markdown: body.slice((match.index ?? 0) + match[0].length, matches[index + 1]?.index ?? body.length).trim(),
  }));
  return { number, slug: String(number), title: heading[1].trim(), sections };
}

export const LEARNING_CHAPTERS: LearningChapter[] = Array.from({ length: 12 }, (_, index) =>
  parseChapter(index + 1, manuscript),
);

const ENGLISH_LEARNING_CHAPTERS: LearningChapter[] = Array.from({ length: 12 }, (_, index) => {
  const chapter = parseChapter(index + 1, englishManuscript, true);
  const original = LEARNING_CHAPTERS[index];
  if (chapter.sections.length !== original.sections.length) {
    throw new Error(`Chapter ${index + 1} translations have different section counts`);
  }
  // Keep saved section links and reading position valid when switching languages.
  return { ...chapter, sections: chapter.sections.map((section, position) => ({
    ...section, id: original.sections[position].id,
  })) };
});

export function getLearningChapters(locale: Locale): readonly LearningChapter[] {
  return locale === "en" ? ENGLISH_LEARNING_CHAPTERS : LEARNING_CHAPTERS;
}

export function learningPath(chapter: number, sectionId?: string): string {
  return `/learn/${chapter}${sectionId ? `#${sectionId}` : ""}`;
}

