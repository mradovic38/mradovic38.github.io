# mradovic38.github.io

A personal website built on [Jon Barron's template](https://github.com/jonbarron/jonbarron_website), with a blog added.
Built with [Jekyll](https://jekyllrb.com) and deployed to GitHub Pages by GitHub Actions.
Posts are Markdown files with LaTeX math, typeset by [MathJax](https://www.mathjax.org).

## Layout

```
index.html           home page: Barron's template, edited by hand   ← edit first
stylesheet.css       Barron's original stylesheet, unmodified
extras.css           additions: dark mode toggle, email link notice, blog post styles, code highlighting
images/              profile photo, project thumbnails and videos, favicons, post figures
_config.yml          name, URL, site settings
_posts/              blog posts, one Markdown file each
blog/index.html      list of all posts
_layouts/            blog page and post templates (same table markup as index.html)
_plugins/math.rb     keeps LaTeX safe from Markdown (see below)
_drafts/             unpublished drafts; f-divergences/scripts_en/ regenerates that post's figures
```

## Editing the home page

Just like the original template, everything is plain HTML in `index.html`:

- **Photo:** `images/profile.jpeg` (square works best; it's cropped to a circle).
- **Bio and links:** edit the text and the `Email / CV / Scholar / Twitter / Github` links. For a CV, add
  `data/CV.pdf` (or change the link).
- **Research entries:** each paper or project is one `<tr>...</tr>`. The Flow Matching Policy entry is highlighted
  (`bgcolor="#ffffd0"`) and plays a looping video on hover; the SpriteFlow entry replays its video from the start on
  each hover (it uses `onmouseenter`/`onmouseleave`, so moving the mouse inside the row doesn't restart it). To add one,
  copy an entry and replace its name (`flowbc` / `spriteflow`) with something unique everywhere in it. Thumbnails are
  160×160.
- **Miscellanea:** the Blog row fills itself in with your latest posts. The Teaching row is an example of Barron's
  colored-box rows, so copy it for talks, service, awards, etc.
- **Favicon:** replace the files in `images/favicon/`.

## Updating the CV

The CV source is [`_drafts/cv/cv.tex`](_drafts/cv/cv.tex) (it must stay one page). After editing, compile it and copy the
PDF to the file the site links to:

```bash
cd _drafts/cv && latexmk -pdf cv.tex && cp cv.pdf ../../data/CV.pdf
```

## Writing a post

Create `_posts/YYYY-MM-DD-short-title.md`:

```markdown
---
layout: post
title: My post title
description: One-line summary shown on the blog page and in RSS.
tags: [ml, notes]
---

Text in **Markdown**, with inline math $a^2 + b^2 = c^2$ and display math:

$$
\int_0^\infty e^{-x^2}\, dx = \frac{\sqrt{\pi}}{2}
$$
```

Commit and push; the site redeploys in about a minute. The URL will be `/blog/YYYY/MM/DD/short-title/`.
[`_drafts/writing-posts-with-math.md`](_drafts/writing-posts-with-math.md) is a reference post showing everything
that's supported. It lives in `_drafts/`, so it isn't published; preview it by adding `--drafts` to the
`jekyll serve` command (it appears on the blog page with today's date). Unfinished posts can go there too.

### Math

Write LaTeX the way you would in a `.tex` file:

| Syntax                                                 | Renders as                          |
| ------------------------------------------------------ | ----------------------------------- |
| `$...$` or `\(...\)`                                   | inline math                         |
| `$$...$$` or `\[...\]`                                 | display math                        |
| `\begin{align}...\end{align}` (also `equation`, `gather`, `multline`) | numbered equations, use `\label` / `\eqref` |
| `\$`                                                   | a literal dollar sign               |

`_plugins/math.rb` pulls math out before Markdown runs and puts it back afterwards, so `_`, `*`, `\\`, `|` and `{{`
inside formulas never get mangled. Math inside code blocks and `inline code` is left alone. A `$` that is followed by
a space or preceded by one (like `$5 and $10`) isn't treated as math.

Custom macros (`\R`, `\E` so far) are in [`_includes/mathjax.html`](_includes/mathjax.html). MathJax only loads on
pages that actually contain math.

## Running locally

Needs Ruby 3.x (macOS's built-in Ruby 2.6 is too old; `brew install ruby`, then follow the PATH hint it prints).

```bash
bundle install
bundle exec jekyll serve --livereload
```

Then open http://localhost:4000. Or, without installing Ruby, use Docker:

```bash
docker run --rm -it -p 4000:4000 -p 35729:35729 -v "$PWD":/site -v jekyll-gems:/usr/local/bundle -w /site ruby:3.3 bash -c "bundle install && bundle exec jekyll serve --host 0.0.0.0 --force_polling --livereload"
```

## Deploying

`.github/workflows/pages.yml` builds and publishes on every push to `main`. One-time setup: in the repo on GitHub, go to
**Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.

GitHub's built-in "Deploy from a branch" mode won't work here, because it doesn't run custom plugins like
`_plugins/math.rb`.

### Custom domain

The site is served at [mihailoradovic.com](https://mihailoradovic.com). The domain is registered at Porkbun, whose DNS has
four `A` records for the bare domain (`185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`) and a
`CNAME` record `www` → `mradovic38.github.io`. On GitHub, the domain is set under **Settings → Pages → Custom domain**,
with **Enforce HTTPS** on. No `CNAME` file is needed, because deployments from GitHub Actions ignore it. The `url` in
`_config.yml` must match the domain.
