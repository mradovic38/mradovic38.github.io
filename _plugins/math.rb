# frozen_string_literal: true

# Lets you write LaTeX in Markdown files exactly as you would in a .tex file:
#
#   inline:   $e^{i\pi} + 1 = 0$        or  \( ... \)
#   display:  $$ \int_0^1 f(x)\,dx $$   or  \[ ... \]
#   envs:     \begin{align} ... \end{align}  (also equation, gather, multline)
#
# Markdown and Liquid would normally mangle math (`_` and `*` become emphasis,
# `\\` becomes `\`, `{{` is a Liquid tag). So before rendering, every math span
# is swapped for an opaque placeholder; after Markdown is converted to HTML,
# the placeholders are swapped back for the original TeX, which MathJax then
# typesets in the browser. Code blocks and inline code are left untouched.
# A literal dollar sign is written as \$.

require "cgi"

module MathPlaceholders
  TOKEN = "MATHPLACEHOLDER%dEND"
  TOKEN_RE = /MATHPLACEHOLDER(\d+)END/

  # Code is matched first and passed through untouched, so `$` in code is safe.
  MATH_RE = /
      (?<code>^[\ \t]*(?<fence>`{3,}|~{3,})[^\n]*\n.*?^[\ \t]*\k<fence>[\ \t]*$) # fenced code
    | (?<code2>(?<ticks>`+)(?!`)(?:(?!\n[\ \t]*\n).)+?(?<!`)\k<ticks>(?!`))    # inline code
    | (?<escaped>\\\$)                                                        # literal \$
    | \$\$(?<display>.+?)\$\$
    | \\\[(?<display2>.+?)\\\]
    | (?<env>\\begin\{(?<name>equation|align|alignat|gather|multline|flalign|eqnarray)\*?\}.*?\\end\{\k<name>\*?\})
    | \\\((?<inline>.+?)\\\)
    | (?<!\\)\$(?![\s$])(?<inline2>[^\n]+?)(?<![\s\\])\$(?!\d)
  /mx

  module_function

  def markdown?(item)
    ext = item.respond_to?(:extname) ? item.extname : item.ext
    exts = item.site.config["markdown_ext"].to_s.split(",").map { |e| ".#{e.strip}" }
    exts.include?(ext.to_s.downcase)
  end

  def protect(item, payload)
    return unless markdown?(item)

    store = []
    item.content = item.content.gsub(MATH_RE) do
      m = Regexp.last_match
      next m[0] if m[:code] || m[:code2] || m[:escaped]

      display = m[:display] || m[:display2]
      inline = m[:inline] || m[:inline2]
      tex =
        if display then "\\[#{display}\\]"
        elsif inline then "\\(#{inline}\\)"
        else m[:env]
        end
      store << { tex: tex, block: inline.nil? }
      format(TOKEN, store.size - 1)
    end

    return if store.empty?

    item.instance_variable_set(:@math_store, store)
    item.data["math"] = true
    payload["page"]["math"] = true if payload["page"].is_a?(Hash)
  end

  def restore(item)
    store = item.instance_variable_get(:@math_store)
    return unless store

    html = item.content
    # A display equation alone in its paragraph becomes a block element.
    html = html.gsub(%r{<p>\s*#{TOKEN_RE}\s*</p>}) do
      entry = store[Regexp.last_match(1).to_i]
      entry[:block] ? %(<div class="math-display">#{CGI.escapeHTML(entry[:tex])}</div>) : Regexp.last_match(0)
    end
    item.content = html.gsub(TOKEN_RE) { CGI.escapeHTML(store[Regexp.last_match(1).to_i][:tex]) }
  end
end

Jekyll::Hooks.register [:documents, :pages], :pre_render do |item, payload|
  MathPlaceholders.protect(item, payload)
end

Jekyll::Hooks.register [:documents, :pages], :post_convert do |item|
  MathPlaceholders.restore(item)
end
