/**
 * 回答文本渲染。
 *
 * 为什么手写而不是引 Markdown 库：
 * 1. 提示词（app/chat/prompt_builder.py）已固定输出结构，
 *    实际只会出现「## 标题 / **加粗** / - 列表 / [1] 引用编号」这几种标记；
 * 2. 引 Markdown 库必须开 dangerouslySetInnerHTML，等于把模型输出当 HTML 执行，
 *    存在注入风险，而这里全部走 React 文本节点，天然安全；
 * 3. 引用编号需要渲染成可点击的上标，交给通用 Markdown 库反而要写插件。
 *
 * 因此这里做的是「恰好够用的轻量解析」，遇到不认识的语法原样当文本输出，
 * 不会因为模型多输出一个符号就渲染崩掉。
 */

import type { ReactNode } from "react";

/** 内联标记：**加粗** 与 [n] 引用编号 */
const INLINE_PATTERN = /(\*\*[^*]+\*\*)|(\[\d+\])/g;
/** 引用编号：形如 [1] */
const CITATION_PATTERN = /^\[(\d+)\]$/;

interface AnswerContentProps {
  text: string;
  /** 点击引用编号时的回调，参数是 1 起的编号 */
  onCitationClick?: (index: number) => void;
}

/** 渲染完整回答 */
export function AnswerContent({ text, onCitationClick }: AnswerContentProps) {
  const blocks = splitBlocks(text);

  return (
    <div className="prose-legal text-sm text-ink-800">
      {blocks.map((block, index) => renderBlock(block, index, onCitationClick))}
    </div>
  );
}

interface Block {
  kind: "heading" | "list" | "paragraph";
  lines: string[];
}

/**
 * 把文本切成块。
 *
 * 连续以 - 或数字加点开头的行合并成一个列表块，
 * 以 ## 开头的行成为标题块，其余连续行合并成段落块。
 */
function splitBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let current: Block | null = null;

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trimEnd();

    if (!line.trim()) {
      current = null;
      continue;
    }

    const isHeading = /^#{2,4}\s+/.test(line);
    const isList = /^([-*]|\d+\.)\s+/.test(line.trim());

    const kind: Block["kind"] = isHeading
      ? "heading"
      : isList
        ? "list"
        : "paragraph";

    if (current && current.kind === kind) {
      current.lines.push(line);
    } else {
      current = { kind, lines: [line] };
      blocks.push(current);
    }
  }

  return blocks;
}

function renderBlock(
  block: Block,
  key: number,
  onCitationClick?: (index: number) => void,
): ReactNode {
  if (block.kind === "heading") {
    return (
      <h3
        key={key}
        className="mt-5 mb-2 font-serif text-base font-medium text-ink-900 first:mt-0"
      >
        {renderInline(block.lines[0].replace(/^#{2,4}\s+/, ""), onCitationClick)}
      </h3>
    );
  }

  if (block.kind === "list") {
    return (
      <ul key={key} className="my-2.5 space-y-1.5">
        {block.lines.map((line, lineIndex) => (
          <li key={lineIndex} className="flex gap-2.5">
            <span
              aria-hidden="true"
              className="mt-[0.6em] h-1 w-1 shrink-0 rounded-full bg-ink-400"
            />
            <span className="min-w-0 flex-1">
              {renderInline(line.trim().replace(/^([-*]|\d+\.)\s+/, ""), onCitationClick)}
            </span>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <p key={key} className="my-2.5 first:mt-0 last:mb-0">
      {block.lines.map((line, lineIndex) => (
        <span key={lineIndex}>
          {lineIndex > 0 ? <br /> : null}
          {renderInline(line, onCitationClick)}
        </span>
      ))}
    </p>
  );
}

/**
 * 渲染行内标记，返回 React 节点数组。
 *
 * 每个片段都用带 key 的 span 包裹，保证 React 能正确做 diff，
 * 流式追加内容时不会整段重排、出现视觉跳动。
 */
function renderInline(
  text: string,
  onCitationClick?: (index: number) => void,
): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(INLINE_PATTERN)) {
    const matchIndex = match.index ?? 0;
    if (matchIndex > lastIndex) {
      nodes.push(
        <span key={`t-${lastIndex}`}>{text.slice(lastIndex, matchIndex)}</span>,
      );
    }

    const token = match[0];

    if (token.startsWith("**")) {
      nodes.push(
        <strong key={`b-${matchIndex}`} className="font-medium text-ink-900">
          {token.slice(2, -2)}
        </strong>,
      );
    } else {
      const citationMatch = CITATION_PATTERN.exec(token);
      const citationIndex = citationMatch ? Number(citationMatch[1]) : 0;
      if (citationIndex > 0) {
        nodes.push(
          <CitationMark
            key={`c-${matchIndex}`}
            index={citationIndex}
            onClick={onCitationClick}
          />,
        );
      } else {
        nodes.push(<span key={`r-${matchIndex}`}>{token}</span>);
      }
    }

    lastIndex = matchIndex + token.length;
  }

  if (lastIndex < text.length) {
    nodes.push(<span key={`t-${lastIndex}`}>{text.slice(lastIndex)}</span>);
  }

  return nodes;
}

/**
 * 引用编号上标。
 *
 * 有对应法源时渲染成按钮并可点击定位；无回调时退化为纯文本，
 * 避免出现「点了没反应」的假按钮。
 */
function CitationMark({
  index,
  onClick,
}: {
  index: number;
  onClick?: (index: number) => void;
}) {
  if (!onClick) {
    return (
      <span className="mx-0.5 align-super text-[0.7em] text-seal-600">
        [{index}]
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={() => onClick(index)}
      aria-label={`查看第 ${index} 条法源`}
      className="mx-0.5 align-super rounded text-[0.7em] font-medium text-seal-600 underline-offset-2 transition-colors duration-150 hover:bg-seal-50 hover:underline"
    >
      [{index}]
    </button>
  );
}
