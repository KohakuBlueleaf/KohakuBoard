// Markdown rendering utility for KohakuBoard
import MarkdownIt from "markdown-it";
import DOMPurify from "isomorphic-dompurify";
import hljs from "highlight.js/lib/core";

// Import languages for code highlighting
import javascript from "highlight.js/lib/languages/javascript";
import typescript from "highlight.js/lib/languages/typescript";
import python from "highlight.js/lib/languages/python";
import bash from "highlight.js/lib/languages/bash";
import json from "highlight.js/lib/languages/json";
import yaml from "highlight.js/lib/languages/yaml";
import sql from "highlight.js/lib/languages/sql";
import shell from "highlight.js/lib/languages/shell";

// Register languages
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("js", javascript);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("ts", typescript);
hljs.registerLanguage("python", python);
hljs.registerLanguage("py", python);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("sh", shell);
hljs.registerLanguage("shell", shell);
hljs.registerLanguage("json", json);
hljs.registerLanguage("yaml", yaml);
hljs.registerLanguage("yml", yaml);
hljs.registerLanguage("sql", sql);

/**
 * Create markdown renderer with syntax highlighting
 */
const md = new MarkdownIt({
  html: true, // Enable HTML tags in markdown
  linkify: true, // Auto-convert URLs to links
  typographer: true, // Smart quotes and typography
  breaks: false, // Don't convert \n to <br> (preserve paragraphs)
  highlight: function (code, lang) {
    // Syntax highlighting with highlight.js
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(code, { language: lang }).value;
      } catch (e) {
        console.error("Highlight error:", e);
      }
    }
    // Fallback to escaped HTML
    return md.utils.escapeHtml(code);
  },
});

/**
 * Render markdown to HTML with XSS protection
 *
 * @param {string} markdown - Raw markdown content
 * @param {Object} options - Rendering options
 * @param {boolean} options.stripFrontmatter - Remove YAML frontmatter (default: true)
 * @returns {string} - Sanitized HTML
 */
export function renderMarkdown(markdown, options = {}) {
  const { stripFrontmatter = true } = options;

  if (!markdown) return "";

  let content = markdown;

  // Strip YAML frontmatter
  if (stripFrontmatter) {
    content = content.replace(/^---[\s\S]*?---\n/, "");
  }

  // Render markdown to HTML
  const rawHTML = md.render(content);

  // Sanitize HTML to prevent XSS
  const sanitizedHTML = DOMPurify.sanitize(rawHTML, {
    ALLOWED_TAGS: [
      "h1",
      "h2",
      "h3",
      "h4",
      "h5",
      "h6",
      "p",
      "br",
      "hr",
      "strong",
      "em",
      "b",
      "i",
      "u",
      "code",
      "pre",
      "a",
      "img",
      "ul",
      "ol",
      "li",
      "table",
      "thead",
      "tbody",
      "tr",
      "th",
      "td",
      "blockquote",
      "div",
      "span",
      "input", // For task lists
    ],
    ALLOWED_ATTR: [
      "href",
      "src",
      "alt",
      "title",
      "class",
      "id",
      "type",
      "checked",
      "disabled", // For checkboxes
      "target",
      "rel", // For links
    ],
  });

  return sanitizedHTML;
}

/**
 * Parse YAML frontmatter from markdown
 *
 * @param {string} markdown - Markdown content with frontmatter
 * @returns {Object} - Parsed frontmatter (title, description, icon, etc.)
 */
export function parseFrontmatter(markdown) {
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n/;
  const match = markdown.match(frontmatterRegex);

  if (!match) {
    return { title: null, description: null, icon: null };
  }

  const yamlLines = match[1].split("\n");
  const meta = {};

  for (const line of yamlLines) {
    const [key, ...valueParts] = line.split(":");
    if (key && valueParts.length) {
      meta[key.trim()] = valueParts.join(":").trim();
    }
  }

  return {
    title: meta.title || null,
    description: meta.description || null,
    icon: meta.icon || null,
  };
}
